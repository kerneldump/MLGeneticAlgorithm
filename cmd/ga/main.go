package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"github.com/aram/MLGeneticAlgorithm/ga"
)

// OneMaxChromosome is a chromosome for the One-Max problem.
type OneMaxChromosome struct {
	Genes []bool
}

// Fitness calculates the fitness of the chromosome.
func (c *OneMaxChromosome) Fitness() float64 {
	score := 0
	for _, gene := range c.Genes {
		if gene {
			score++
		}
	}
	return float64(score)
}

// Crossover creates a new chromosome by combining the genes of two parents.
func (c *OneMaxChromosome) Crossover(other ga.Chromosome) ga.Chromosome {
	parent1 := c.Genes
	parent2 := other.(*OneMaxChromosome).Genes
	crossoverPoint := rand.Intn(len(parent1))
	childGenes := make([]bool, len(parent1))
	copy(childGenes[:crossoverPoint], parent1[:crossoverPoint])
	copy(childGenes[crossoverPoint:], parent2[crossoverPoint:])
	return &OneMaxChromosome{Genes: childGenes}
}

// Mutate randomly flips a gene in the chromosome.
func (c *OneMaxChromosome) Mutate() {
	mutationPoint := rand.Intn(len(c.Genes))
	c.Genes[mutationPoint] = !c.Genes[mutationPoint]
}

func main() {
	rand.Seed(time.Now().UnixNano())

	example := flag.String("example", "onemax", "The example to run (onemax or tsp)")
	flag.Parse()

	switch *example {
	case "onemax":
		runOneMax()
	case "tsp":
		runTSP()
	default:
		log.Fatalf("Unknown example: %s", *example)
	}
}

func runOneMax() {
	// Create an initial population.
	population := make([]ga.Chromosome, 100)
	for i := range population {
		genes := make([]bool, 20)
		for j := range genes {
			genes[j] = rand.Float64() < 0.5
		}
		population[i] = &OneMaxChromosome{Genes: genes}
	}

	// Create a new genetic algorithm.
	geneticAlgorithm := ga.New(
		ga.WithPopulation(population),
		ga.WithMutationRate(0.01),
		ga.WithCrossoverRate(0.8),
		ga.WithGenerations(100),
		ga.WithElitism(true),
	)

	// Run the genetic algorithm.
	if err := geneticAlgorithm.Run(); err != nil {
		log.Fatalf("Failed to run genetic algorithm: %v", err)
	}

	// Print the best chromosome.
	best := geneticAlgorithm.Best()
	fmt.Printf("Best chromosome fitness: %v\n", best.Fitness())
}

func runTSP() {
	// Load cities from CSV.
	cities, err := loadCities("examples/tsp.csv")
	if err != nil {
		log.Fatalf("Failed to load cities: %v", err)
	}

	// Validate minimum number of cities
	if len(cities) < 2 {
		log.Fatalf("Need at least 2 cities for TSP, got %d", len(cities))
	}

	fmt.Printf("Loaded %d cities for TSP\n", len(cities))

	// Create an initial population.
	population := make([]ga.Chromosome, 100)
	for i := range population {
		route := make([]ga.City, len(cities))
		copy(route, cities)
		rand.Shuffle(len(route), func(i, j int) {
			route[i], route[j] = route[j], route[i]
		})
		population[i] = &ga.TSPChromosome{Route: route}
	}

	fmt.Println("Running genetic algorithm...")

	// Create a new genetic algorithm.
	geneticAlgorithm := ga.New(
		ga.WithPopulation(population),
		ga.WithMutationRate(0.02),
		ga.WithCrossoverRate(0.85),
		ga.WithGenerations(200),
		ga.WithElitism(true),
		ga.WithProgressCallback(func(generation int, best ga.Chromosome) {
			// Print progress every 20 generations
			if generation%20 == 0 || generation == 199 {
				fmt.Printf("Generation %d: Best fitness = %.6f\n", generation, best.Fitness())
			}
		}),
	)

	// Run the genetic algorithm.
	if err := geneticAlgorithm.Run(); err != nil {
		log.Fatalf("Failed to run genetic algorithm: %v", err)
	}

	// Get the best route.
	best := geneticAlgorithm.Best().(*ga.TSPChromosome)

	// Calculate and display the total distance
	totalDistance := 0.0
	for i := 0; i < len(best.Route)-1; i++ {
		dx := best.Route[i].X - best.Route[i+1].X
		dy := best.Route[i].Y - best.Route[i+1].Y
		totalDistance += math.Sqrt(dx*dx + dy*dy)
	}
	// Add distance back to start
	dx := best.Route[len(best.Route)-1].X - best.Route[0].X
	dy := best.Route[len(best.Route)-1].Y - best.Route[0].Y
	totalDistance += math.Sqrt(dx*dx + dy*dy)

	fmt.Printf("Best route fitness: %v (total distance: %.2f)\n", best.Fitness(), totalDistance)

	// Visualize the best route.
	err = ga.VisualizeTSP(best.Route, "tsp_route.svg")
	if err != nil {
		log.Fatalf("Failed to visualize TSP route: %v", err)
	}

	fmt.Println("TSP route visualization saved to tsp_route.svg")
}

func loadCities(filename string) ([]ga.City, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %w", filename, err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}

	if len(records) < 2 { // At least header + 1 data row
		return nil, fmt.Errorf("CSV file must contain at least a header and one data row")
	}

	cities := make([]ga.City, 0, len(records)-1)
	for i, record := range records {
		if i == 0 { // Skip header
			continue
		}

		if len(record) < 3 {
			return nil, fmt.Errorf("row %d: expected at least 3 columns (name, x, y), got %d", i+1, len(record))
		}

		name := record[0]
		if name == "" {
			return nil, fmt.Errorf("row %d: city name cannot be empty", i+1)
		}

		x, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			return nil, fmt.Errorf("row %d: invalid x coordinate '%s': %w", i+1, record[1], err)
		}

		y, err := strconv.ParseFloat(record[2], 64)
		if err != nil {
			return nil, fmt.Errorf("row %d: invalid y coordinate '%s': %w", i+1, record[2], err)
		}

		cities = append(cities, ga.City{Name: name, X: x, Y: y})
	}

	return cities, nil
}