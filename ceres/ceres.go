// Package ceres implements Graphite's Ceres database format
package ceres

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type Options struct {
	Sparse       bool
	FLock        bool
	VerboseError bool
	TimeNow      func() time.Time
}

func unitMultiplier(s string) (int64, error) {
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
	return 0, fmt.Errorf("invalid unit multiplier [%v]", s)
}

var retentionRegexp *regexp.Regexp = regexp.MustCompile("^(\\d+)([smhdwy]+)$")

func parseRetentionPart(retentionPart string) (int64, error) {
	part, err := strconv.ParseInt(retentionPart, 10, 32)
	if err == nil {
		return part, nil
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
	return multiplier * value, err
}

// ParseRetentionDef parses a retention definition as you would find in the storage-schemas.conf of a Carbon installation.
// Note that this only parses a single retention definition, if you have multiple definitions (separated by a comma)
// you will have to split them yourself.
//
// ParseRetentionDef("10s:14d") Retention{10, 120960}
//
// See: http://graphite.readthedocs.org/en/1.0/config-carbon.html#storage-schemas-conf
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
		return nil, fmt.Errorf("Failed to parse Points: %v", err)
	}
	points /= precision

	return &Retention{
		SecondsPerPoint: precision,
		Points:          points,
	}, err
}

// ParseRetentionDefs parses a retention definitions as you would find in the storage-schemas.conf of a Carbon installation.
// Note that this parses even multiple definitions
//
// ParseRetentionDefs("10s:1d,60s:14d") []Retention{{10, 86400}, {60, 120960}}
//
// See: http://graphite.readthedocs.org/en/1.0/config-carbon.html#storage-schemas-conf
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

// Retention levels describe a given archive in the database. How detailed it is and how far back it records.
type Retention struct {
	SecondsPerPoint int64
	Points          int64
}

func (retention *Retention) MarshalJSON() ([]byte, error) {
	var ret [2]int64
	ret[0] = retention.SecondsPerPoint
	ret[1] = retention.Points
	return json.Marshal(ret)
}

func (retention *Retention) UnmarshalJSON(data []byte) error {
	var ret [2]int64
	err := json.Unmarshal(data, &ret)
	if err != nil {
		return err
	}
	retention.SecondsPerPoint = ret[0]
	retention.Points = ret[1]
	if retention.Points < 0 {
		return fmt.Errorf("amount of Points can't be negative: %v", retention.Points)
	}
	if retention.SecondsPerPoint < 0 {
		return fmt.Errorf("time step can't be negative: %v", retention.Points)
	}
	return nil
}

func (retention *Retention) MaxRetention() int64 {
	return int64(retention.SecondsPerPoint) * int64(retention.Points)
}

func (retention *Retention) MaxPoints() int64 {
	return retention.Points
}

// NewRetention creates a new retention structure (see description above)
func NewRetention(secondsPerPoint, numberOfPoints int64) Retention {
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

// Metadata represents a metadata for a Ceres database file.
type Metadata struct {
	AggregationMethod AggregationMethod `json:"aggregationMethod"`
	TimeStep          int               `json:"timeStep"`
	XFilesFactor      float32           `json:"xFilesFactor"`
	Retentions        Retentions
}

type CeresSlice struct {
	Filename        string
	StartTime       int64
	SecondsPerPoint int64
	Points          int64

	file *os.File
}

func (slice *CeresSlice) MaxRetention() int64 {
	return slice.StartTime + slice.SecondsPerPoint*slice.Points
}

func (slice *CeresSlice) MinRetention() int64 {
	return slice.StartTime
}

func (slice *CeresSlice) Step() int64 {
	return slice.SecondsPerPoint
}

func (slice *CeresSlice) getReader() io.ReaderAt {
	if slice.file == nil {
		slice.file, _ = os.Open(slice.Filename)
	}
	return slice.file
}

func (slice *CeresSlice) Read(from, until, step int64, method AggregationMethod) ([]float64, error) {
	val, err := slice.readSlice(slice.getReader(), from, until, step, method)
	// We don't want to have too many file descriptors open
	if slice.file != nil {
		_ = slice.file.Close()
		slice.file = nil
	}
	return val, err
}

func (slice *CeresSlice) readSlice(reader io.ReaderAt, from, until, step int64, aggregationMethod AggregationMethod) ([]float64, error) {
	if from < slice.StartTime {
		from = slice.StartTime
	}
	if until > slice.MaxRetention() {
		until = slice.MaxRetention()
	}
	if step == -1 {
		step = slice.SecondsPerPoint
	}
	if until < from {
		return nil, ErrInvalidFrom
	}

	fromRead := from - mod(from, slice.SecondsPerPoint)
	untilRead := until - mod(until, slice.SecondsPerPoint)
	fromResult := from - mod(from, step)
	untilResult := until - mod(until, step)
	res := make([]float64, (untilResult-fromResult+1)/step)
	readBuffer := make([]byte, PointSize*(untilRead-fromRead+1)/slice.SecondsPerPoint)
	stepConversionRatio := step / slice.SecondsPerPoint
	aggregationBuffer := make([]float64, stepConversionRatio)

	offset := fromRead - slice.StartTime
	_, err := reader.ReadAt(readBuffer, offset)
	if err != nil && !errors.Is(err, io.EOF) {
		return nil, err
	}
	begin := int64(0)
	resultIdx := 0
	if stepConversionRatio == 1 {
		for {
			end := begin + Float64Size
			if end > int64(len(readBuffer)) {
				break
			}
			res[resultIdx] = unpackFloat64(readBuffer[begin:end])
			resultIdx++
			begin = end
		}
	} else {
		for {
			end := begin + stepConversionRatio*Float64Size
			if end > int64(len(readBuffer)) {
				break
			}
			for i := int64(0); i < stepConversionRatio; i++ {
				aggregationBuffer[i] = unpackFloat64(readBuffer[begin+i*Float64Size : begin+i*Float64Size+Float64Size])
			}
			res[resultIdx] = aggregate(aggregationMethod, aggregationBuffer)
			resultIdx++
			begin = end
		}
	}

	return res, nil
}

/*
func (archive *archiveInfo) Offset() int64 {
	return -1
}

func (archive *archiveInfo) PointOffset(baseInterval, interval int) int64 {
	timeDistance := interval - baseInterval
	pointDistance := timeDistance / archive.SecondsPerPoint
	byteDistance := pointDistance * PointSize
	myOffset := archive.Offset() + int64(mod(byteDistance, archive.Size()))

	return myOffset
}

func (archive *archiveInfo) End() int64 {
	return archive.Offset() + int64(archive.Size())
}

func (archive *archiveInfo) Interval(time int) int {
	return time - mod(time, archive.SecondsPerPoint) + archive.SecondsPerPoint
}
*/

type Ceres struct {
	TimeNow      func() time.Time
	file         string
	metadataFile *os.File

	// Metadata
	metadata Metadata
	archives map[int][]*CeresSlice
}

type CeresOption struct {
	path     *string
	sparse   *bool
	flock    *bool
	metadata *Metadata
	time     func() time.Time
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

// Create a new Whisper database file and write its header.
func Create(vaArgs ...*CeresOption) (*Ceres, error) {
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
		if arg.time != nil {
			opts.time = arg.time
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
	if opts.time == nil {
		opts.time = func() time.Time {
			return time.Now()
		}
	}

	if opts.path == nil {
		return nil, fmt.Errorf("path can't be empty")
	}
	if opts.metadata == nil {
		return nil, fmt.Errorf("metadata can't be empty")
	}

	sort.Sort(retentionsByPrecision{opts.metadata.Retentions})
	if err := validateRetentions(opts.metadata.Retentions); err != nil {
		return nil, err
	}

	path := *opts.path

	// For compatibility reason remove trailing `.wsp` from Filename
	// This is required as package should look like a drop-in replacement for go-whisper in terms of API
	idx := strings.Index(path, ".wsp")
	if idx > 0 {
		path = path[:idx]
	}

	metadataFile, err := os.Create(path + "/" + metadataFile)
	if err != nil {
		return nil, err
	}

	if *opts.flock {
		if err = syscall.Flock(int(metadataFile.Fd()), syscall.LOCK_EX); err != nil {
			_ = metadataFile.Close()
			return nil, err
		}
	}

	ceres := &Ceres{
		file:         path,
		metadataFile: metadataFile,
		metadata:     *opts.metadata,
	}

	d, err := json.Marshal(ceres.metadata)
	if err != nil {
		_ = metadataFile.Close()
		return nil, err
	}

	_, err = metadataFile.Write(d)
	if err != nil {
		_ = metadataFile.Close()
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
		if !(retention.SecondsPerPoint < nextRetention.SecondsPerPoint) {
			return fmt.Errorf("ceres database may not be configured having two archives with the same precision (archive%v: %v, archive%v: %v)", i, retention, i+1, nextRetention)
		}

		if mod(nextRetention.SecondsPerPoint, retention.SecondsPerPoint) != 0 {
			return fmt.Errorf("higher precision archives' precision must evenly divide all lower precision archives' precision (archive%v: %v, archive%v: %v)", i, retention.SecondsPerPoint, i+1, nextRetention.SecondsPerPoint)
		}

		if retention.MaxRetention() >= nextRetention.MaxRetention() {
			return fmt.Errorf("lower precision archives must cover larger time intervals than higher precision archives (archive%v: %v seconds, archive%v: %v seconds)", i, retention.MaxRetention(), i+1, nextRetention.MaxRetention())
		}
	}
	return nil
}

// Open an existing Whisper database and read it's header
func Open(path string) (*Ceres, error) {
	return OpenWithOptions(path, &Options{
		FLock: false,
	})
}

func OpenWithOptions(path string, options *Options) (*Ceres, error) {
	metadataPath := filepath.Clean(path + "/" + metadataFile)
	metadataFile, err := os.OpenFile(metadataPath, os.O_RDWR, 0666)
	if err != nil {
		return nil, err
	}

	if options.FLock {
		if err = syscall.Flock(int(metadataFile.Fd()), syscall.LOCK_EX); err != nil {
			_ = metadataFile.Close()
			return nil, err
		}
	}

	ceres := Ceres{
		file:         filepath.Clean(path),
		metadataFile: metadataFile,
	}

	if options.TimeNow != nil {
		ceres.TimeNow = options.TimeNow
	} else {
		ceres.TimeNow = func() time.Time {
			return time.Now()
		}
	}

	metadataDecoder := json.NewDecoder(metadataFile)
	err = metadataDecoder.Decode(&ceres.metadata)
	if err != nil {
		_ = metadataFile.Close()
		return nil, err
	}

	slices, err := filepath.Glob(ceres.file + "/*.slice")
	if err != nil {
		_ = metadataFile.Close()
		return nil, err
	}

	ceres.archives = make(map[int][]*CeresSlice)
	for _, slice := range slices {
		stat, err := os.Stat(slice)
		if err != nil {
			if options.VerboseError {
				fmt.Printf("error getting information about slice '%v': %v", slice, err)
			}
			continue
		}
		name := filepath.Base(slice[:len(slice)-len(".slice")])
		parts := strings.Split(name, "@")
		if len(parts) != 2 {
			if options.VerboseError {
				fmt.Printf("file '%v' is malformed, expected format: 'StartTime@retention.slice', got: '%v'", name+".slice", parts)
			}
			// Ignoring file with broken file name
			continue
		}
		step, err := strconv.ParseInt(parts[1], 10, 64)
		if err != nil {
			if options.VerboseError {
				fmt.Printf("failed to parse step '%v' as integer: %v", parts[1], err)
			}
			// Ignoring file with broken file name
			continue
		}
		startTime, err := strconv.ParseInt(parts[0], 10, 64)
		if err != nil {
			if options.VerboseError {
				fmt.Printf("failed to parse StartTime '%v' as integer: %v", parts[0], err)
			}
			// Ignoring file with broken file name
			continue
		}
		info := CeresSlice{
			Filename:        slice,
			StartTime:       startTime,
			SecondsPerPoint: step,
			Points:          stat.Size() / PointSize,
		}
		_, ok := ceres.archives[int(step)]
		if ok {
			ceres.archives[int(step)] = append(ceres.archives[int(step)], &info)
			continue
		}
		ceres.archives[int(step)] = []*CeresSlice{&info}
	}

	return &ceres, nil
}

// Close the ceres file
func (ceres *Ceres) Close() {
	_ = ceres.metadataFile.Close()
	for _, archives := range ceres.archives {
		for _, archive := range archives {
			if archive.file != nil {
				_ = archive.file.Close()
			}
		}
	}
}

// ArchiveCount returns total amount of retentions found
func (ceres *Ceres) ArchiveCount() int {
	return len(ceres.archives)
}

// ArchivesInfo returns information about archives
func (ceres *Ceres) ArchivesInfo() map[int][]*CeresSlice {
	return ceres.archives
}

// Size calculates total number of bytes the Ceres file should be according to the metadata.
func (ceres *Ceres) Size() (int64, error) {
	size, err := ceres.MetadataSize()
	if err != nil {
		return 0, err
	}
	for _, archive := range ceres.archives {
		for _, slice := range archive {
			size += slice.Points * PointSize
		}
	}
	return size, nil
}

// MetadataSize -  returns size of metadata file
func (ceres *Ceres) MetadataSize() (int64, error) {
	if ceres.metadataFile == nil {
		return -1, fmt.Errorf("metadata file is not defined, this shouldn't happen")
	}
	stat, err := ceres.metadataFile.Stat()
	if err != nil {
		return 0, err
	}
	return stat.Size(), nil
}

// AggregationMethod - returns string representation of aggregation method
func (ceres *Ceres) AggregationMethodString() string {
	return ceres.metadata.AggregationMethod.String()
}

// AggregationMethod - returns string representation of aggregation method
func (ceres *Ceres) AggregationMethod() AggregationMethod {
	return ceres.metadata.AggregationMethod
}

// MaxRetention - returns max retnetion in seconds
func (ceres *Ceres) MaxRetention() int64 {
	return ceres.metadata.Retentions[0].MaxRetention()
}

// StartTime - calculate the starting time for a whisper db.
func (ceres *Ceres) StartTime() int64 {
	now := ceres.TimeNow().Unix() // TODO: danger of 2030 something overflow
	return now - ceres.MaxRetention()
}

// XFilesFactor returns configured xFilesFactor from metadata
func (ceres *Ceres) XFilesFactor() float32 {
	return ceres.metadata.XFilesFactor
}

// Retentions - returns list of retentions for specific path
func (ceres *Ceres) Retentions() []Retention {
	return ceres.metadata.Retentions
}

func isInInterval(archiveIntervalStartTime, archiveIntervalEndTime, intervalStartTime, intervalEndTime int64) bool {
	return intervalStartTime <= archiveIntervalEndTime && archiveIntervalStartTime <= intervalEndTime
}

// Fetch a TimeSeries for a given time span from the file.
func (ceres *Ceres) Fetch(fromTime, untilTime int64) (timeSeries *TimeSeries, err error) {
	now := ceres.TimeNow().Unix() // TODO: danger of 2030 something overflow
	if fromTime > untilTime {
		// TODO: use errors
		return nil, ErrInvalidFrom
	}
	oldestTime := ceres.StartTime()
	// range is in the future
	// TODO: think about breaking compatibility and allow to read future datapoints
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

	// Get list of all archives that needs to be read
	archivesToRead := make([]*CeresSlice, 0)
	stepTime := int64(-1)
	for _, archives := range ceres.archives {
		for _, archive := range archives {
			archiveStartTime := archive.MinRetention()
			// Assumption: our start time is wi
			if archiveStartTime > untilTime {
				break
			}
			archiveStopTime := archive.MaxRetention()
			if isInInterval(archiveStartTime, archiveStopTime, fromTime, untilTime) {
				archivesToRead = append(archivesToRead, archive)
				if archive.Step() > stepTime {
					stepTime = archive.Step()
				}
			}
		}
	}

	res := TimeSeries{
		fromTime:  int(fromTime),
		untilTime: int(untilTime),
		step:      int(stepTime),
		values:    make([]float64, fromTime-untilTime),
	}

	/*
		base := fromTime
		for _, archive := range archivesToRead {

		}
	*/

	return &res, nil

	/*
		fromInterval := archive.Interval(fromTime)
		untilInterval := archive.Interval(untilTime)
		baseInterval := ceres.getBaseInterval(archive)

		if baseInterval == 0 {
			step := archive.SecondsPerPoint
			Points := (untilInterval - fromInterval) / step
			values := make([]float64, Points)
			for i := range values {
				values[i] = math.NaN()
			}
			return &TimeSeries{fromInterval, untilInterval, step, values}, nil
		}

		// Zero-length time range: always include the next point
		if fromInterval == untilInterval {
			untilInterval += archive.SecondsPerPoint
		}

		fromOffset := archive.PointOffset(baseInterval, fromInterval)
		untilOffset := archive.PointOffset(baseInterval, untilInterval)

		series, err := ceres.readSeries(fromOffset, untilOffset, archive)
		if err != nil {
			return nil, err
		}

		values := make([]float64, len(series))
		for i := range values {
			values[i] = math.NaN()
		}
		currentInterval := fromInterval
		step := archive.SecondsPerPoint

		for i, dPoint := range series {
			if dPoint.interval == currentInterval {
				values[i] = dPoint.value
			}
			currentInterval += step
		}

		return &TimeSeries{fromInterval, untilInterval, step, values}, nil
	*/
}

/*

// CheckEmpty - checks if TimeSeries have a Points for a given time span from the file.
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

/*
// Update a value in the database.
// If the timestamp is in the future or outside of the maximum retention it will
// fail immediately.
func (whisper *Ceres) Update(value float64, timestamp int) (err error) {
	// recover panics and return as error
	defer func() {
		if e := recover(); e != nil {
			err = errors.New(e.(string))
		}
	}()

	diff := int(time.Now().Unix()) - timestamp
	if !(diff < whisper.MaxRetention() && diff >= 0) {
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

	myInterval := timestamp - mod(timestamp, archive.SecondsPerPoint)
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

func reversePoints(Points []*TimeSeriesPoint) {
	size := len(Points)
	end := size / 2

	for i := 0; i < end; i++ {
		Points[i], Points[size-i-1] = Points[size-i-1], Points[i]
	}
}

func (whisper *Ceres) UpdateMany(Points []*TimeSeriesPoint) (err error) {
	// recover panics and return as error
	defer func() {
		if e := recover(); e != nil {
			err = errors.New(e.(string))
		}
	}()

	// sort the Points, newest first
	reversePoints(Points)
	sort.Stable(timeSeriesPointsNewestFirst{Points})

	now := int(time.Now().Unix()) // TODO: danger of 2030 something overflow

	var currentPoints []*TimeSeriesPoint
	for _, archive := range whisper.archives {
		currentPoints, Points = extractPoints(Points, now, archive.MaxRetention())
		if len(currentPoints) == 0 {
			continue
		}
		// reverse currentPoints
		reversePoints(currentPoints)
		err = whisper.archiveUpdateMany(archive, currentPoints)
		if err != nil {
			return
		}

		if len(Points) == 0 { // nothing left to do
			break
		}
	}
	return
}

func (whisper *Ceres) archiveUpdateMany(archive *archiveInfo, Points []*TimeSeriesPoint) error {
	alignedPoints := alignPoints(archive, Points)
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
			interval := point.interval - mod(point.interval, lower.SecondsPerPoint)
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

func extractPoints(Points []*TimeSeriesPoint, now int, maxRetention int) (currentPoints []*TimeSeriesPoint, remainingPoints []*TimeSeriesPoint) {
	maxAge := now - maxRetention
	for i, point := range Points {
		if point.Time < maxAge {
			if i > 0 {
				return Points[:i-1], Points[i-1:]
			} else {
				return []*TimeSeriesPoint{}, Points
			}
		}
	}
	return Points, remainingPoints
}

func alignPoints(archive *archiveInfo, Points []*TimeSeriesPoint) []dataPoint {
	alignedPoints := make([]dataPoint, 0, len(Points))
	positions := make(map[int]int)
	for _, point := range Points {
		dPoint := dataPoint{point.Time - mod(point.Time, archive.SecondsPerPoint), point.Value}
		if p, ok := positions[dPoint.interval]; ok {
			alignedPoints[p] = dPoint
		} else {
			alignedPoints = append(alignedPoints, dPoint)
			positions[dPoint.interval] = len(alignedPoints) - 1
		}
	}
	return alignedPoints
}

func packSequences(archive *archiveInfo, Points []dataPoint) (intervals []int, packedBlocks [][]byte) {
	intervals = make([]int, 0)
	packedBlocks = make([][]byte, 0)
	for i, point := range Points {
		if i == 0 || point.interval != intervals[len(intervals)-1]+archive.SecondsPerPoint {
			intervals = append(intervals, point.interval)
			packedBlocks = append(packedBlocks, point.Bytes())
		} else {
			packedBlocks[len(packedBlocks)-1] = append(packedBlocks[len(packedBlocks)-1], point.Bytes()...)
		}
	}
	return
}

// getPointOffset - calculate the offset for a given interval in an archive
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
		if lower.SecondsPerPoint > archive.SecondsPerPoint {
			return whisper.archives[i:]
		}
	}
	return
}

func (whisper *Ceres) propagate(timestamp int, higher, lower *archiveInfo) (bool, error) {
	lowerIntervalStart := timestamp - mod(timestamp, lower.SecondsPerPoint)

	higherFirstOffset := whisper.getPointOffset(lowerIntervalStart, higher)

	// TODO: extract all this series extraction stuff
	higherPoints := lower.SecondsPerPoint / higher.SecondsPerPoint
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
		currentInterval += higher.SecondsPerPoint
	}

	// propagate aggregateValue to propagate from neighborValues if we have enough known Points
	if len(knownValues) == 0 {
		return false, nil
	}
	knownPercent := float32(len(knownValues)) / float32(len(series))
	if knownPercent < whisper.xFilesFactor { // check we have enough data Points to propagate a value
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

func (whisper *Ceres) readInt(offset int64) (int, error) {
	// TODO: make errors better
	b := make([]byte, IntSize)
	_, err := whisper.file.ReadAt(b, offset)
	if err != nil {
		return 0, err
	}

	return unpackInt(b), nil
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
*/
