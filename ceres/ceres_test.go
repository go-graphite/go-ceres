package ceres

import (
	"encoding/binary"
	"math"
	"testing"
)

func TestIsInterval(t *testing.T) {
	tests := []struct {
		name           string
		fromRequest    int64
		untilRequest   int64
		fromArchive    int64
		untilArchive   int64
		expectedResult bool
	}{
		{
			name:           "Starts before, no match",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    1,
			untilArchive:   9,
			expectedResult: false,
		},
		{
			name:           "Starts before, within",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    1,
			untilArchive:   14,
			expectedResult: true,
		},
		{
			name:           "Starts before, partial",
			fromRequest:    10,
			untilRequest:   14,
			fromArchive:    1,
			untilArchive:   12,
			expectedResult: true,
		},
		{
			name:           "Starts after, within",
			fromRequest:    10,
			untilRequest:   20,
			fromArchive:    12,
			untilArchive:   25,
			expectedResult: true,
		},
		{
			name:           "Starts after, partial",
			fromRequest:    10,
			untilRequest:   20,
			fromArchive:    12,
			untilArchive:   15,
			expectedResult: true,
		},
		{
			name:           "Starts after, no match",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    14,
			untilArchive:   16,
			expectedResult: false,
		},
		{
			name:           "Starts exact, exact",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    10,
			untilArchive:   12,
			expectedResult: true,
		},
		{
			name:           "Starts exact, one point",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    10,
			untilArchive:   10,
			expectedResult: true,
		},
		{
			name:           "Ends exact, one point",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    12,
			untilArchive:   12,
			expectedResult: true,
		},
		{
			name:           "Starts within, one point",
			fromRequest:    10,
			untilRequest:   12,
			fromArchive:    11,
			untilArchive:   11,
			expectedResult: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			res := isInInterval(test.fromArchive, test.untilArchive, test.fromRequest, test.untilRequest)
			if res != test.expectedResult {
				t.Errorf("Unexpected result for test '%v', got %v, expected %v", test.name, res, test.expectedResult)
			}
		})
	}
}

type mockReaderAt struct {
	content []byte
}

func (r *mockReaderAt) ReadAt(p []byte, off int64) (n int, err error) {
	i := 0
	for i = range p {
		if i >= len(r.content) {
			break
		}
		p[i] = r.content[i]
	}
	return i, nil
}

func floatSliceToBytes(in []float64) []byte {
	res := make([]byte, Float64Size*len(in))
	for i := 0; i < Float64Size*len(in); i += Float64Size {
		binary.BigEndian.PutUint64(res[i:i+Float64Size], math.Float64bits(in[i/Float64Size]))
	}

	return res
}

func TestSliceRead(t *testing.T) {
	tests := []struct {
		testName  string
		startTime int64
		sliceStep int64
		values    []float64

		readFrom           int64
		readUntil          int64
		requestStep        int64
		aggregationMethond AggregationMethod

		expectedErr    error
		expectedResult []float64
	}{
		{
			testName:  "simple, one sec, 4 values",
			startTime: 1,
			sliceStep: 1,
			values:    []float64{1, 2, 3, 4},

			readFrom:           1,
			readUntil:          4,
			requestStep:        1,
			aggregationMethond: Average,

			expectedErr:    nil,
			expectedResult: []float64{1, 2, 3, 4},
		},
		{
			testName:  "4 values, requestStep=2, aggregation=Avg",
			startTime: 1,
			sliceStep: 1,
			values:    []float64{8, 4, 2, 0},

			readFrom:           1,
			readUntil:          4,
			requestStep:        2,
			aggregationMethond: Average,

			expectedErr:    nil,
			expectedResult: []float64{6, 1},
		},
		{
			testName:  "4 values, requestStep=2, aggregation=Max",
			startTime: 1,
			sliceStep: 1,
			values:    []float64{8, 4, 2, 0},

			readFrom:           1,
			readUntil:          4,
			requestStep:        2,
			aggregationMethond: Max,

			expectedErr:    nil,
			expectedResult: []float64{8, 2},
		},
	}
	for _, test := range tests {
		t.Run(test.testName, func(t *testing.T) {
			slice := CeresSlice{
				Filename:        "test.slice",
				StartTime:       test.startTime,
				SecondsPerPoint: test.sliceStep,
				Points:          int64(len(test.values)),
			}
			r := &mockReaderAt{
				content: floatSliceToBytes(test.values),
			}
			res, err := slice.readSlice(r, test.readFrom, test.readUntil, test.requestStep, test.aggregationMethond)
			if err != test.expectedErr {
				t.Errorf("unexpected err, got %+v, expected %+v", test.expectedErr, err)
			}
			if len(res) != len(test.expectedResult) {
				t.Errorf("unexpected length, got %v, expected %v, returned data %+v", len(res), len(test.expectedResult), res)
				t.FailNow()
			}

			for i, v := range res {
				if v != test.expectedResult[i] {
					t.Errorf("unexpected value at pos=%v, got %v, expected %v", i, v, test.expectedResult[i])
				}
			}
		})
	}
}
