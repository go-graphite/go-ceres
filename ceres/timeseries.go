package ceres

import (
	"fmt"
	"time"
)

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
	Points := make([]TimeSeriesPoint, len(ts.values))
	for i, value := range ts.values {
		Points[i] = TimeSeriesPoint{Time: ts.fromTime + ts.step*i, Value: value}
	}
	return Points
}

func (ts *TimeSeries) String() string {
	return fmt.Sprintf("TimeSeries{'%v' '%-v' %v %v}", time.Unix(int64(ts.fromTime), 0), time.Unix(int64(ts.untilTime), 0), ts.step, ts.values)
}

type TimeSeriesPoint struct {
	Time  int
	Value float64
}
