package ceres

import (
	"encoding/binary"
	"math"
)

//	Implementation of modulo that works like Python
//	Thanks @timmow for this
func mod(a, b int64) int64 {
	return a - (b * int64(math.Floor(float64(a)/float64(b))))
}

// Unpacks float
func unpackFloat64(b []byte) float64 {
	return math.Float64frombits(binary.BigEndian.Uint64(b))
}
