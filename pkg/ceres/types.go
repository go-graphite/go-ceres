package ceres

import (
	"encoding/json"
	"fmt"
)

var errUnknownAggregatiomMethodFmt = "unknown aggregation method %v, supported %v"

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
