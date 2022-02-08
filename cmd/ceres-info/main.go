package main

import (
	"flag"
	"log"

	"github.com/go-graphite/go-ceres/pkg/ceres"
)

func main() {
	filename := flag.String("n", "", "file to get information about")

	flag.Parse()

	if filename == nil || *filename == "" {
		log.Fatalln("Filename must not be empty!")
	}
	log.Println("Priting file info", *filename)

	c, err := ceres.OpenWithOptions(*filename, &ceres.Options{
		FLock:        false,
		VerboseError: true,
	})

	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	size, err := c.Size()
	if err != nil {
		log.Printf("failed to get information about size: %v", err)
	}

	archives := make(map[int][]ceres.SliceInfo)
	for step, infos := range c.ArchivesInfo() {
		archives[step] = []ceres.SliceInfo{}
		for _, info := range infos {
			archives[step] = append(archives[step], *info)
		}
	}

	log.Printf("ArchiveCount: %v\n"+
		"MaxRetention: %v\n"+
		"Size: %v\n"+
		"XFilesFactor: %v\n"+
		"AggregationMethod: %v\n"+
		"StartTime: %v\n"+
		"Retentions: %+v\n"+
		"ArchivesInfo: %+v\n",
		c.ArchiveCount(),
		c.MaxRetention(),
		size,
		c.XFilesFactor(),
		c.AggregationMethod(),
		c.StartTime(),
		c.Retentions(),
		archives,
	)
}
