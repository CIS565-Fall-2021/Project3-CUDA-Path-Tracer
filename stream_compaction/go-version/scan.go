package main

import (
	"fmt"
)

func ilog2ceil(x int) int {
	if x == 1 {
		return 0
	}
	lg := 0
	for x -= 1; x != 0; x >>= 1 {
		lg++
	}
	return 1 + lg
}

func scan_iter(k int, d int, x []int, y []int, done chan int) {
	if k >= (1 << d) {
		y[k] = x[k] + x[k-(1<<d)]
	} else {
		y[k] = x[k]
	}
	done <- 0
}

func naive_scan(n int, arr []int) {
	x := arr
	y := make([]int, n)
	c := make(chan int)
	for d := 0;  d < ilog2ceil(n); d++ {
		for k := 0; k < n; k++ {
			go scan_iter(k, d, x, y, c)
		}
		for k := 0; k < n; k++ {
			<-c
		}
		fmt.Printf("source: %v\tdest: %v\n", x, y)
		x, y = y, x
	}
	arr = y
}


func main() {
	x := []int{1, 5, 0, 1, 2, 0, 3, 0, 1}
	fmt.Print(len(x), x, "\n")
	naive_scan(len(x), x)
	fmt.Print(len(x), x, "\n")
}

