/*
	Package whisper implements Graphite's Whisper database format
*/
package ceres

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	IntSize         = 4
	FloatSize       = 4
	Float64Size     = 8
	PointSize       = 12
	MetadataSize    = 16
	ArchiveInfoSize = 12
)

const (
	Seconds = 1
	Minutes = 60
	Hours   = 3600
	Days    = 86400
	Weeks   = 86400 * 7
	Years   = 86400 * 365
)

type AggregationMethod int

const (
	Average AggregationMethod = iota + 1
	Sum
	Last
	Max
	Min
)

var supportedAggregationMethods = []string{"average", "sum", "last", "max", "min"}

var aggregationMethodToString = map[AggregationMethod]string{
	Average: "average",
	Sum:     "sum",
	Last:    "last",
	Max:     "max",
	Min:     "min",
}

var stringToAggregationMethod = map[string]AggregationMethod{
	"average": Average,
	"sum":     Sum,
	"last":    Last,
	"max":     Max,
	"min":     Min,
}

var errUnknownAggregatiomMethodFmt = "unknown aggregation method %v, supported %v"

func (a *AggregationMethod) MarshalJSON() ([]byte, error) {
	if s, ok := aggregationMethodToString[*a]; ok {
		return json.Marshal(s)
	}
	return nil, fmt.Errorf(errUnknownAggregatiomMethodFmt, a, supportedAggregationMethods)
}

func (a *AggregationMethod) UnmarshalJSON(data []byte) error {
	var s string
	err := json.Unmarshal(data, &s)
	if err != nil {
		return err
	}
	var ok bool
	*a, ok = stringToAggregationMethod[s]
	if !ok {
		return fmt.Errorf(errUnknownAggregatiomMethodFmt, s, supportedAggregationMethods)
	}
	return nil
}

type Options struct {
	Sparse bool
	FLock  bool
}

func unitMultiplier(s string) (int, error) {
	switch {
	case strings.HasPrefix(s, "s"):
		return Seconds, nil
	case strings.HasPrefix(s, "m"):
		return Minutes, nil
	case strings.HasPrefix(s, "h"):
		return Hours, nil
	case strings.HasPrefix(s, "d"):
		return Days, nil
	case strings.HasPrefix(s, "w"):
		return Weeks, nil
	case strings.HasPrefix(s, "y"):
		return Years, nil
	}
	return 0, fmt.Errorf("Invalid unit multiplier [%v]", s)
}

var retentionRegexp *regexp.Regexp = regexp.MustCompile("^(\\d+)([smhdwy]+)$")

func parseRetentionPart(retentionPart string) (int, error) {
	part, err := strconv.ParseInt(retentionPart, 10, 32)
	if err == nil {
		return int(part), nil
	}
	if !retentionRegexp.MatchString(retentionPart) {
		return 0, fmt.Errorf("%v", retentionPart)
	}
	matches := retentionRegexp.FindStringSubmatch(retentionPart)
	value, err := strconv.ParseInt(matches[1], 10, 32)
	if err != nil {
		panic(fmt.Sprintf("Regex on %v is borked, %v cannot be parsed as int", retentionPart, matches[1]))
	}
	multiplier, err := unitMultiplier(matches[2])
	return multiplier * int(value), err
}

/*
  ParseRetentionDef parses a retention definition as you would find in the storage-schemas.conf of a Carbon install.
  Note that this only parses a single retention definition, if you have multiple definitions (separated by a comma)
  you will have to split them yourself.

  ParseRetentionDef("10s:14d") Retention{10, 120960}

  See: http://graphite.readthedocs.org/en/1.0/config-carbon.html#storage-schemas-conf
*/
func ParseRetentionDef(retentionDef string) (*Retention, error) {
	parts := strings.Split(retentionDef, ":")
	if len(parts) != 2 {
		return nil, fmt.Errorf("Not enough parts in retentionDef [%v]", retentionDef)
	}
	precision, err := parseRetentionPart(parts[0])
	if err != nil {
		return nil, fmt.Errorf("Failed to parse precision: %v", err)
	}

	points, err := parseRetentionPart(parts[1])
	if err != nil {
		return nil, fmt.Errorf("Failed to parse points: %v", err)
	}
	points /= precision

	return &Retention{
		SecondsPerPoint: precision,
		Points:          points,
	}, err
}

func ParseRetentionDefs(retentionDefs string) (Retentions, error) {
	retentions := make(Retentions, 0)
	for _, retentionDef := range strings.Split(retentionDefs, ",") {
		retention, err := ParseRetentionDef(retentionDef)
		if err != nil {
			return nil, err
		}
		retentions = append(retentions, *retention)
	}
	return retentions, nil
}

/*
  A retention level.

  Retention levels describe a given archive in the database. How detailed it is and how far back
  it records.
*/
type Retention struct {
	SecondsPerPoint int
	Points          int
}

func (r *Retention) MarshalJSON() ([]byte, error) {
	var ret [2]int
	ret[0] = r.SecondsPerPoint
	ret[1] = r.Points
	return json.Marshal(ret)
}

func (r *Retention) UnmarshalJSON(data []byte) error {
	var ret [2]int
	err := json.Unmarshal(data, &ret)
	if err != nil {
		return err
	}
	r.SecondsPerPoint = ret[0]
	r.Points = ret[1]
	if r.Points < 0 {
		return fmt.Errorf("amount of points can't be negative: %v", r.Points)
	}
	if r.SecondsPerPoint < 0 {
		return fmt.Errorf("time step can't be negative: %v", r.Points)
	}
	return nil
}

func (retention *Retention) MaxRetention() int {
	return retention.SecondsPerPoint * retention.Points
}

func NewRetention(secondsPerPoint, numberOfPoints int) Retention {
	return Retention{
		SecondsPerPoint: secondsPerPoint,
		Points:          numberOfPoints,
	}
}

type Retentions []Retention

func (r Retentions) Len() int {
	return len(r)
}

func (r Retentions) Swap(i, j int) {
	r[i], r[j] = r[j], r[i]
}

type retentionsByPrecision struct{ Retentions }

func (r retentionsByPrecision) Less(i, j int) bool {
	return r.Retentions[i].SecondsPerPoint < r.Retentions[j].SecondsPerPoint
}

/*
	Represents a Ceres database file.
*/
type Metadata struct {
	AggregationMethod AggregationMethod `json:"aggregationMethod"`
	TimeStep          int               `json:"timeStep"`
	XFilesFactor      float32           `json:"xFilesFactor"`
	Retentions        Retentions
}

type sliceInfo struct {
	startTime int
	step      int
	size      int
}

type archiveInfo struct {
	filename        string
	startTime       int
	secondsPerPoint int

	slices []sliceInfo
}

/*
func (archive *archiveInfo) Offset() int64 {
	return -1
}

func (archive *archiveInfo) PointOffset(baseInterval, interval int) int64 {
	timeDistance := interval - baseInterval
	pointDistance := timeDistance / archive.secondsPerPoint
	byteDistance := pointDistance * PointSize
	myOffset := archive.Offset() + int64(mod(byteDistance, archive.Size()))

	return myOffset
}

func (archive *archiveInfo) End() int64 {
	return archive.Offset() + int64(archive.Size())
}

func (archive *archiveInfo) Interval(time int) int {
	return time - mod(time, archive.secondsPerPoint) + archive.secondsPerPoint
}
*/

type Ceres struct {
	file string

	// Metadata
	metadata Metadata
	archives []*archiveInfo
}

// Wrappers for whisper.file operations
func (whisper *Ceres) fileWriteAt(b []byte, off int64) error {
	_, err := whisper.file.WriteAt(b, off)
	return err
}

// Wrappers for file.ReadAt operations
func (whisper *Ceres) fileReadAt(b []byte, off int64) error {
	_, err := whisper.file.ReadAt(b, off)
	return err
}

type CeresOption struct {
	path     *string
	sparse   *bool
	flock    *bool
	metadata *Metadata
}

func WithSparse() *CeresOption {
	v := true
	return &CeresOption{
		sparse: &v,
	}
}

func WithPath(path string) *CeresOption {
	return &CeresOption{
		path: &path,
	}
}

func WithFlock() *CeresOption {
	v := true
	return &CeresOption{
		flock: &v,
	}
}

func WithMetadata(m *Metadata) *CeresOption {
	return &CeresOption{
		metadata: m,
	}
}

const (
	metadataFile = ".ceres-node"
)

/*
	Create a new Whisper database file and write it's header.
*/
func Create(vaArgs ...*CeresOption) (whisper *Ceres, err error) {
	opts := CeresOption{}
	for _, arg := range vaArgs {
		if arg.flock != nil {
			opts.flock = arg.flock
		}
		if arg.sparse != nil {
			opts.sparse = arg.sparse
		}
		if arg.path != nil {
			opts.path = arg.path
		}
		if opts.metadata != nil {
			return nil, fmt.Errorf("please define only one metadata")
		}
		if arg.metadata != nil {
			opts.metadata = arg.metadata
		}
	}

	if opts.flock == nil {
		v := false
		opts.flock = &v
	}
	if opts.sparse == nil {
		v := false
		opts.sparse = &v
	}

	if opts.path == nil {
		return nil, fmt.Errorf("path can't be empty")
	}
	if opts.metadata == nil {
		return nil, fmt.Errorf("metadata can't be empty")
	}

	sort.Sort(retentionsByPrecision{opts.metadata.Retentions})
	if err = validateRetentions(opts.metadata.Retentions); err != nil {
		return nil, err
	}

	path := *opts.path

	idx := strings.Index(path, ".wsp")
	if idx > 0 {
		path = path[:idx]
	}

	file, err := os.Create(path + "/" + metadataFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	if *opts.flock {
		if err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX); err != nil {
			return nil, err
		}
	}

	ceres := &Ceres{
		file:     path,
		metadata: *opts.metadata,
	}

	d, err := json.Marshal(ceres.metadata)
	if err != nil {
		return nil, err
	}

	_, err = file.Write(d)
	if err != nil {
		return nil, err
	}
	return ceres, nil
}

func validateRetentions(retentions Retentions) error {
	if len(retentions) == 0 {
		return fmt.Errorf("No retentions")
	}
	for i, retention := range retentions {
		if i == len(retentions)-1 {
			break
		}

		nextRetention := retentions[i+1]
		if !(retention.secondsPerPoint < nextRetention.secondsPerPoint) {
			return fmt.Errorf("A Whisper database may not be configured having two archives with the same precision (archive%v: %v, archive%v: %v)", i, retention, i+1, nextRetention)
		}

		if mod(nextRetention.secondsPerPoint, retention.secondsPerPoint) != 0 {
			return fmt.Errorf("Higher precision archives' precision must evenly divide all lower precision archives' precision (archive%v: %v, archive%v: %v)", i, retention.secondsPerPoint, i+1, nextRetention.secondsPerPoint)
		}

		if retention.MaxRetention() >= nextRetention.MaxRetention() {
			return fmt.Errorf("Lower precision archives must cover larger time intervals than higher precision archives (archive%v: %v seconds, archive%v: %v seconds)", i, retention.MaxRetention(), i+1, nextRetention.MaxRetention())
		}

		if retention.numberOfPoints < (nextRetention.secondsPerPoint / retention.secondsPerPoint) {
			return fmt.Errorf("Each archive must have at least enough points to consolidate to the next archive (archive%v consolidates %v of archive%v's points but it has only %v total points)", i+1, nextRetention.secondsPerPoint/retention.secondsPerPoint, i, retention.numberOfPoints)
		}
	}
	return nil
}

/*
  Open an existing Whisper database and read it's header
*/
func Open(path string) (whisper *Ceres, err error) {
	return OpenWithOptions(path, &Options{
		FLock: false,
	})
}

func OpenWithOptions(path string, options *Options) (whisper *Ceres, err error) {
	file, err := os.OpenFile(path, os.O_RDWR, 0666)
	if err != nil {
		return
	}

	defer func() {
		if err != nil {
			whisper = nil
			file.Close()
		}
	}()

	if options.FLock {
		if err = syscall.Flock(int(file.Fd()), syscall.LOCK_EX); err != nil {
			return
		}
	}

	whisper = new(Ceres)
	whisper.file = file

	offset := 0

	// read the metadata
	b := make([]byte, MetadataSize)
	readed, err := file.Read(b)

	if err != nil {
		err = fmt.Errorf("Unable to read header: %s", err.Error())
		return
	}
	if readed != MetadataSize {
		err = fmt.Errorf("Unable to read header: EOF")
		return
	}

	a := unpackInt(b[offset : offset+IntSize])
	if a > 1024 { // support very old format. File starts with lastUpdate and has only average aggregation method
		whisper.aggregationMethod = Average
	} else {
		whisper.aggregationMethod = AggregationMethod(a)
	}
	offset += IntSize
	whisper.maxRetention = unpackInt(b[offset : offset+IntSize])
	offset += IntSize
	whisper.xFilesFactor = unpackFloat32(b[offset : offset+FloatSize])
	offset += FloatSize
	archiveCount := unpackInt(b[offset : offset+IntSize])
	offset += IntSize

	// read the archive info
	b = make([]byte, ArchiveInfoSize)

	whisper.archives = make([]*archiveInfo, 0)
	for i := 0; i < archiveCount; i++ {
		readed, err = file.Read(b)
		if err != nil || readed != ArchiveInfoSize {
			err = fmt.Errorf("Unable to read archive %d metadata", i)
			return
		}
		whisper.archives = append(whisper.archives, unpackArchiveInfo(b))
	}

	return whisper, nil
}

func (whisper *Ceres) writeHeader() (err error) {
	b := make([]byte, whisper.MetadataSize())
	i := 0
	i += packInt(b, int(whisper.aggregationMethod), i)
	i += packInt(b, whisper.maxRetention, i)
	i += packFloat32(b, whisper.xFilesFactor, i)
	i += packInt(b, len(whisper.archives), i)
	for _, archive := range whisper.archives {
		i += packInt(b, archive.offset, i)
		i += packInt(b, archive.secondsPerPoint, i)
		i += packInt(b, archive.numberOfPoints, i)
	}
	_, err = whisper.file.Write(b)

	return err
}

/*
  Close the whisper file
*/
func (whisper *Ceres) Close() {
	whisper.file.Close()
}

/*
  Calculate the total number of bytes the Whisper file should be according to the metadata.
*/
func (whisper *Ceres) Size() int {
	size := whisper.MetadataSize()
	for _, archive := range whisper.archives {
		size += archive.Size()
	}
	return size
}

/*
  Calculate the number of bytes the metadata section will be.
*/
func (whisper *Ceres) MetadataSize() int {
	return MetadataSize + (ArchiveInfoSize * len(whisper.archives))
}

/* Return aggregation method */
func (whisper *Ceres) AggregationMethod() string {
	aggr := "unknown"
	switch whisper.aggregationMethod {
	case Average:
		aggr = "Average"
	case Sum:
		aggr = "Sum"
	case Last:
		aggr = "Last"
	case Max:
		aggr = "Max"
	case Min:
		aggr = "Min"
	}
	return aggr
}

/* Return max retention in seconds */
func (whisper *Ceres) MaxRetention() int {
	return whisper.maxRetention
}

/* Return xFilesFactor */
func (whisper *Ceres) XFilesFactor() float32 {
	return whisper.xFilesFactor
}

/* Return retentions */
func (whisper *Ceres) Retentions() []Retention {
	ret := make([]Retention, 0, 4)
	for _, archive := range whisper.archives {
		ret = append(ret, archive.Retention)
	}

	return ret
}

/*
  Update a value in the database.

  If the timestamp is in the future or outside of the maximum retention it will
  fail immediately.
*/
func (whisper *Ceres) Update(value float64, timestamp int) (err error) {
	// recover panics and return as error
	defer func() {
		if e := recover(); e != nil {
			err = errors.New(e.(string))
		}
	}()

	diff := int(time.Now().Unix()) - timestamp
	if !(diff < whisper.maxRetention && diff >= 0) {
		return fmt.Errorf("Timestamp not covered by any archives in this database")
	}
	var archive *archiveInfo
	var lowerArchives []*archiveInfo
	var i int
	for i, archive = range whisper.archives {
		if archive.MaxRetention() < diff {
			continue
		}
		lowerArchives = whisper.archives[i+1:] // TODO: investigate just returning the positions
		break
	}

	myInterval := timestamp - mod(timestamp, archive.secondsPerPoint)
	point := dataPoint{myInterval, value}

	_, err = whisper.file.WriteAt(point.Bytes(), whisper.getPointOffset(myInterval, archive))
	if err != nil {
		return err
	}

	higher := archive
	for _, lower := range lowerArchives {
		propagated, err := whisper.propagate(myInterval, higher, lower)
		if err != nil {
			return err
		} else if !propagated {
			break
		}
		higher = lower
	}

	return nil
}

func reversePoints(points []*TimeSeriesPoint) {
	size := len(points)
	end := size / 2

	for i := 0; i < end; i++ {
		points[i], points[size-i-1] = points[size-i-1], points[i]
	}
}

func (whisper *Ceres) UpdateMany(points []*TimeSeriesPoint) (err error) {
	// recover panics and return as error
	defer func() {
		if e := recover(); e != nil {
			err = errors.New(e.(string))
		}
	}()

	// sort the points, newest first
	reversePoints(points)
	sort.Stable(timeSeriesPointsNewestFirst{points})

	now := int(time.Now().Unix()) // TODO: danger of 2030 something overflow

	var currentPoints []*TimeSeriesPoint
	for _, archive := range whisper.archives {
		currentPoints, points = extractPoints(points, now, archive.MaxRetention())
		if len(currentPoints) == 0 {
			continue
		}
		// reverse currentPoints
		reversePoints(currentPoints)
		err = whisper.archiveUpdateMany(archive, currentPoints)
		if err != nil {
			return
		}

		if len(points) == 0 { // nothing left to do
			break
		}
	}
	return
}

func (whisper *Ceres) archiveUpdateMany(archive *archiveInfo, points []*TimeSeriesPoint) error {
	alignedPoints := alignPoints(archive, points)
	intervals, packedBlocks := packSequences(archive, alignedPoints)

	baseInterval := whisper.getBaseInterval(archive)
	if baseInterval == 0 {
		baseInterval = intervals[0]
	}

	for i := range intervals {
		myOffset := archive.PointOffset(baseInterval, intervals[i])
		bytesBeyond := int(myOffset-archive.End()) + len(packedBlocks[i])
		if bytesBeyond > 0 {
			pos := len(packedBlocks[i]) - bytesBeyond
			err := whisper.fileWriteAt(packedBlocks[i][:pos], myOffset)
			if err != nil {
				return err
			}
			err = whisper.fileWriteAt(packedBlocks[i][pos:], archive.Offset())
			if err != nil {
				return err
			}
		} else {
			err := whisper.fileWriteAt(packedBlocks[i], myOffset)
			if err != nil {
				return err
			}
		}
	}

	higher := archive
	lowerArchives := whisper.lowerArchives(archive)

	for _, lower := range lowerArchives {
		seen := make(map[int]bool)
		propagateFurther := false
		for _, point := range alignedPoints {
			interval := point.interval - mod(point.interval, lower.secondsPerPoint)
			if !seen[interval] {
				if propagated, err := whisper.propagate(interval, higher, lower); err != nil {
					panic("Failed to propagate")
				} else if propagated {
					propagateFurther = true
				}
			}
		}
		if !propagateFurther {
			break
		}
		higher = lower
	}
	return nil
}

func extractPoints(points []*TimeSeriesPoint, now int, maxRetention int) (currentPoints []*TimeSeriesPoint, remainingPoints []*TimeSeriesPoint) {
	maxAge := now - maxRetention
	for i, point := range points {
		if point.Time < maxAge {
			if i > 0 {
				return points[:i-1], points[i-1:]
			} else {
				return []*TimeSeriesPoint{}, points
			}
		}
	}
	return points, remainingPoints
}

func alignPoints(archive *archiveInfo, points []*TimeSeriesPoint) []dataPoint {
	alignedPoints := make([]dataPoint, 0, len(points))
	positions := make(map[int]int)
	for _, point := range points {
		dPoint := dataPoint{point.Time - mod(point.Time, archive.secondsPerPoint), point.Value}
		if p, ok := positions[dPoint.interval]; ok {
			alignedPoints[p] = dPoint
		} else {
			alignedPoints = append(alignedPoints, dPoint)
			positions[dPoint.interval] = len(alignedPoints) - 1
		}
	}
	return alignedPoints
}

func packSequences(archive *archiveInfo, points []dataPoint) (intervals []int, packedBlocks [][]byte) {
	intervals = make([]int, 0)
	packedBlocks = make([][]byte, 0)
	for i, point := range points {
		if i == 0 || point.interval != intervals[len(intervals)-1]+archive.secondsPerPoint {
			intervals = append(intervals, point.interval)
			packedBlocks = append(packedBlocks, point.Bytes())
		} else {
			packedBlocks[len(packedBlocks)-1] = append(packedBlocks[len(packedBlocks)-1], point.Bytes()...)
		}
	}
	return
}

/*
	Calculate the offset for a given interval in an archive

	This method retrieves the baseInterval and the
*/
func (whisper *Ceres) getPointOffset(start int, archive *archiveInfo) int64 {
	baseInterval := whisper.getBaseInterval(archive)
	if baseInterval == 0 {
		return archive.Offset()
	}
	return archive.PointOffset(baseInterval, start)
}

func (whisper *Ceres) getBaseInterval(archive *archiveInfo) int {
	baseInterval, err := whisper.readInt(archive.Offset())
	if err != nil {
		panic("Failed to read baseInterval")
	}
	return baseInterval
}

func (whisper *Ceres) lowerArchives(archive *archiveInfo) (lowerArchives []*archiveInfo) {
	for i, lower := range whisper.archives {
		if lower.secondsPerPoint > archive.secondsPerPoint {
			return whisper.archives[i:]
		}
	}
	return
}

func (whisper *Ceres) propagate(timestamp int, higher, lower *archiveInfo) (bool, error) {
	lowerIntervalStart := timestamp - mod(timestamp, lower.secondsPerPoint)

	higherFirstOffset := whisper.getPointOffset(lowerIntervalStart, higher)

	// TODO: extract all this series extraction stuff
	higherPoints := lower.secondsPerPoint / higher.secondsPerPoint
	higherSize := higherPoints * PointSize
	relativeFirstOffset := higherFirstOffset - higher.Offset()
	relativeLastOffset := int64(mod(int(relativeFirstOffset+int64(higherSize)), higher.Size()))
	higherLastOffset := relativeLastOffset + higher.Offset()

	series, err := whisper.readSeries(higherFirstOffset, higherLastOffset, higher)
	if err != nil {
		return false, err
	}

	// and finally we construct a list of values
	knownValues := make([]float64, 0, len(series))
	currentInterval := lowerIntervalStart

	for _, dPoint := range series {
		if dPoint.interval == currentInterval {
			knownValues = append(knownValues, dPoint.value)
		}
		currentInterval += higher.secondsPerPoint
	}

	// propagate aggregateValue to propagate from neighborValues if we have enough known points
	if len(knownValues) == 0 {
		return false, nil
	}
	knownPercent := float32(len(knownValues)) / float32(len(series))
	if knownPercent < whisper.xFilesFactor { // check we have enough data points to propagate a value
		return false, nil
	} else {
		aggregateValue := aggregate(whisper.aggregationMethod, knownValues)
		point := dataPoint{lowerIntervalStart, aggregateValue}
		if _, err := whisper.file.WriteAt(point.Bytes(), whisper.getPointOffset(lowerIntervalStart, lower)); err != nil {
			return false, err
		}
	}
	return true, nil
}

func (whisper *Ceres) readSeries(start, end int64, archive *archiveInfo) ([]dataPoint, error) {
	var b []byte
	if start < end {
		b = make([]byte, end-start)
		err := whisper.fileReadAt(b, start)
		if err != nil {
			return nil, err
		}
	} else {
		b = make([]byte, archive.End()-start)
		err := whisper.fileReadAt(b, start)
		if err != nil {
			return nil, err
		}
		b2 := make([]byte, end-archive.Offset())
		err = whisper.fileReadAt(b2, archive.Offset())
		if err != nil {
			return nil, err
		}
		b = append(b, b2...)
	}
	return unpackDataPoints(b), nil
}

func (whisper *Ceres) checkSeriesEmpty(start, end int64, archive *archiveInfo, fromTime, untilTime int) (bool, error) {
	if start < end {
		len := end - start
		return whisper.checkSeriesEmptyAt(start, len, fromTime, untilTime)
	}
	len := archive.End() - start
	empty, err := whisper.checkSeriesEmptyAt(start, len, fromTime, untilTime)
	if err != nil || !empty {
		return empty, err
	}
	return whisper.checkSeriesEmptyAt(archive.Offset(), end-archive.Offset(), fromTime, untilTime)

}

func (whisper *Ceres) checkSeriesEmptyAt(start, len int64, fromTime, untilTime int) (bool, error) {
	b1 := make([]byte, 4)
	// Read first point
	err := whisper.fileReadAt(b1, start)
	if err != nil {
		return false, err
	}
	pointTime := unpackInt(b1)
	if pointTime > fromTime && pointTime < untilTime {
		return false, nil
	}

	b2 := make([]byte, 4)
	// Read last point
	err = whisper.fileReadAt(b2, len-12)
	if err != nil {
		return false, err
	}
	pointTime = unpackInt(b1)
	if pointTime > fromTime && pointTime < untilTime {
		return false, nil
	}
	return true, nil
}

/*
  Calculate the starting time for a whisper db.
*/
func (whisper *Ceres) StartTime() int {
	now := int(time.Now().Unix()) // TODO: danger of 2030 something overflow
	return now - whisper.maxRetention
}

/*
  Fetch a TimeSeries for a given time span from the file.
*/
func (whisper *Ceres) Fetch(fromTime, untilTime int) (timeSeries *TimeSeries, err error) {
	now := int(time.Now().Unix()) // TODO: danger of 2030 something overflow
	if fromTime > untilTime {
		return nil, fmt.Errorf("Invalid time interval: from time '%d' is after until time '%d'", fromTime, untilTime)
	}
	oldestTime := whisper.StartTime()
	// range is in the future
	if fromTime > now {
		return nil, nil
	}
	// range is beyond retention
	if untilTime < oldestTime {
		return nil, nil
	}
	if fromTime < oldestTime {
		fromTime = oldestTime
	}
	if untilTime > now {
		untilTime = now
	}

	// TODO: improve this algorithm it's ugly
	diff := now - fromTime
	var archive *archiveInfo
	for _, archive = range whisper.archives {
		if archive.MaxRetention() >= diff {
			break
		}
	}

	fromInterval := archive.Interval(fromTime)
	untilInterval := archive.Interval(untilTime)
	baseInterval := whisper.getBaseInterval(archive)

	if baseInterval == 0 {
		step := archive.secondsPerPoint
		points := (untilInterval - fromInterval) / step
		values := make([]float64, points)
		for i := range values {
			values[i] = math.NaN()
		}
		return &TimeSeries{fromInterval, untilInterval, step, values}, nil
	}

	// Zero-length time range: always include the next point
	if fromInterval == untilInterval {
		untilInterval += archive.SecondsPerPoint()
	}

	fromOffset := archive.PointOffset(baseInterval, fromInterval)
	untilOffset := archive.PointOffset(baseInterval, untilInterval)

	series, err := whisper.readSeries(fromOffset, untilOffset, archive)
	if err != nil {
		return nil, err
	}

	values := make([]float64, len(series))
	for i := range values {
		values[i] = math.NaN()
	}
	currentInterval := fromInterval
	step := archive.secondsPerPoint

	for i, dPoint := range series {
		if dPoint.interval == currentInterval {
			values[i] = dPoint.value
		}
		currentInterval += step
	}

	return &TimeSeries{fromInterval, untilInterval, step, values}, nil
}

/*
  Check a TimeSeries has a points for a given time span from the file.
*/
func (whisper *Ceres) CheckEmpty(fromTime, untilTime int) (exist bool, err error) {
	now := int(time.Now().Unix()) // TODO: danger of 2030 something overflow
	if fromTime > untilTime {
		return true, fmt.Errorf("Invalid time interval: from time '%d' is after until time '%d'", fromTime, untilTime)
	}
	oldestTime := whisper.StartTime()
	// range is in the future
	if fromTime > now {
		return true, nil
	}
	// range is beyond retention
	if untilTime < oldestTime {
		return true, nil
	}
	if fromTime < oldestTime {
		fromTime = oldestTime
	}
	if untilTime > now {
		untilTime = now
	}

	// TODO: improve this algorithm it's ugly
	diff := now - fromTime
	var archive *archiveInfo
	for _, archive = range whisper.archives {
		fromInterval := archive.Interval(fromTime)
		untilInterval := archive.Interval(untilTime)
		baseInterval := whisper.getBaseInterval(archive)

		if baseInterval == 0 {
			return true, nil
		}

		// Zero-length time range: always include the next point
		if fromInterval == untilInterval {
			untilInterval += archive.SecondsPerPoint()
		}

		fromOffset := archive.PointOffset(baseInterval, fromInterval)
		untilOffset := archive.PointOffset(baseInterval, untilInterval)

		empty, err := whisper.checkSeriesEmpty(fromOffset, untilOffset, archive, fromTime, untilTime)
		if err != nil || !empty {
			return empty, err
		}
		if archive.MaxRetention() >= diff {
			break
		}
	}
	return true, nil
}

func (whisper *Ceres) readInt(offset int64) (int, error) {
	// TODO: make errors better
	b := make([]byte, IntSize)
	_, err := whisper.file.ReadAt(b, offset)
	if err != nil {
		return 0, err
	}

	return unpackInt(b), nil
}

type TimeSeries struct {
	fromTime  int
	untilTime int
	step      int
	values    []float64
}

func (ts *TimeSeries) FromTime() int {
	return ts.fromTime
}

func (ts *TimeSeries) UntilTime() int {
	return ts.untilTime
}

func (ts *TimeSeries) Step() int {
	return ts.step
}

func (ts *TimeSeries) Values() []float64 {
	return ts.values
}

func (ts *TimeSeries) Points() []TimeSeriesPoint {
	points := make([]TimeSeriesPoint, len(ts.values))
	for i, value := range ts.values {
		points[i] = TimeSeriesPoint{Time: ts.fromTime + ts.step*i, Value: value}
	}
	return points
}

func (ts *TimeSeries) String() string {
	return fmt.Sprintf("TimeSeries{'%v' '%-v' %v %v}", time.Unix(int64(ts.fromTime), 0), time.Unix(int64(ts.untilTime), 0), ts.step, ts.values)
}

type TimeSeriesPoint struct {
	Time  int
	Value float64
}

type timeSeriesPoints []*TimeSeriesPoint

func (p timeSeriesPoints) Len() int {
	return len(p)
}

func (p timeSeriesPoints) Swap(i, j int) {
	p[i], p[j] = p[j], p[i]
}

type timeSeriesPointsNewestFirst struct {
	timeSeriesPoints
}

func (p timeSeriesPointsNewestFirst) Less(i, j int) bool {
	return p.timeSeriesPoints[i].Time > p.timeSeriesPoints[j].Time
}

type dataPoint struct {
	interval int
	value    float64
}

func (point *dataPoint) Bytes() []byte {
	b := make([]byte, PointSize)
	packInt(b, point.interval, 0)
	packFloat64(b, point.value, IntSize)
	return b
}

func sum(values []float64) float64 {
	result := 0.0
	for _, value := range values {
		result += value
	}
	return result
}

func aggregate(method AggregationMethod, knownValues []float64) float64 {
	switch method {
	case Average:
		return sum(knownValues) / float64(len(knownValues))
	case Sum:
		return sum(knownValues)
	case Last:
		return knownValues[len(knownValues)-1]
	case Max:
		max := knownValues[0]
		for _, value := range knownValues {
			if value > max {
				max = value
			}
		}
		return max
	case Min:
		min := knownValues[0]
		for _, value := range knownValues {
			if value < min {
				min = value
			}
		}
		return min
	}
	panic("Invalid aggregation method")
}

func packInt(b []byte, v, i int) int {
	binary.BigEndian.PutUint32(b[i:i+IntSize], uint32(v))
	return IntSize
}

func packFloat32(b []byte, v float32, i int) int {
	binary.BigEndian.PutUint32(b[i:i+FloatSize], math.Float32bits(v))
	return FloatSize
}

func packFloat64(b []byte, v float64, i int) int {
	binary.BigEndian.PutUint64(b[i:i+Float64Size], math.Float64bits(v))
	return Float64Size
}

func unpackInt(b []byte) int {
	return int(binary.BigEndian.Uint32(b))
}

func unpackFloat32(b []byte) float32 {
	return math.Float32frombits(binary.BigEndian.Uint32(b))
}

func unpackFloat64(b []byte) float64 {
	return math.Float64frombits(binary.BigEndian.Uint64(b))
}

func unpackArchiveInfo(b []byte) *archiveInfo {
	return &archiveInfo{Retention{unpackInt(b[IntSize : IntSize*2]), unpackInt(b[IntSize*2 : IntSize*3])}, unpackInt(b[:IntSize])}
}

func unpackDataPoint(b []byte) dataPoint {
	return dataPoint{unpackInt(b[0:IntSize]), unpackFloat64(b[IntSize:PointSize])}
}

func unpackDataPoints(b []byte) (series []dataPoint) {
	series = make([]dataPoint, 0, len(b)/PointSize)
	for i := 0; i < len(b); i += PointSize {
		series = append(series, unpackDataPoint(b[i:i+PointSize]))
	}
	return
}

/*
	Implementation of modulo that works like Python
	Thanks @timmow for this
*/
func mod(a, b int) int {
	return a - (b * int(math.Floor(float64(a)/float64(b))))
}
