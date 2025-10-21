# Genetic Algorithm

A flexible and extensible Genetic Algorithm library in Go with examples for optimization problems.

## Genetic Algorithms in Machine Learning

Genetic Algorithms (GAs) are a class of optimization algorithms inspired by the process of natural selection. They are commonly used to find approximate solutions to difficult-to-solve problems. In machine learning, GAs can be used for various tasks, such as:

- **Hyperparameter Tuning:** Optimizing the parameters of a machine learning model (e.g., learning rate, number of layers in a neural network).
- **Feature Selection:** Identifying the most relevant features in a dataset to improve model performance.
- **Neural Architecture Search:** Automatically designing the architecture of a neural network.
- **Combinatorial Optimization:** Solving problems like the Traveling Salesman Problem (TSP).

This project provides a framework for building and running genetic algorithms in Go. It is designed to be flexible, allowing users to define their own chromosomes, fitness functions, and selection methods.

## Features

- **Extensible Framework:** Easily define your own genetic algorithm components.
- **Interfaces:** Core components are defined by interfaces, allowing for custom implementations.
- **Built-in Examples:** Includes One-Max problem and Traveling Salesman Problem (TSP) implementations.
- **Visualization:** SVG generation for TSP route visualization with arrows and city labels.
- **Tournament Selection:** Configurable tournament selection algorithm.
- **CLI:** A simple command-line interface to run example algorithms.

## Requirements

- Go 1.21+

## Install

- Library: `go get github.com/aram/MLGeneticAlgorithm/ga`
- CLI: `go build -o bin/ga ./cmd/ga` or `make build`

## Quickstart (CLI)

Run the examples:

### One-Max Problem
```bash
make example-run
# or
./bin/ga --example=onemax
```

### Traveling Salesman Problem (TSP)
```bash
make example-run-tsp
# or
./bin/ga --example=tsp
```

Note: The TSP example requires a CSV file at `examples/tsp.csv` with the format:
```csv
name,x,y
City1,10.5,20.3
City2,15.2,25.8
City3,8.1,12.7
```

## Library (Go)

### One-Max Example
```go
package main

import (
	"fmt"
	"math/rand"
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

// Clone creates a deep copy of the chromosome.
func (c *OneMaxChromosome) Clone() ga.Chromosome {
	genes := make([]bool, len(c.Genes))
	copy(genes, c.Genes)
	return &OneMaxChromosome{Genes: genes}
}

func main() {
	rand.Seed(time.Now().UnixNano())

	// Create an initial population.
	population := make([]ga.Chromosome, 100)
	for i := range population {
		genes := make([]bool, 10)
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
	geneticAlgorithm.Run()

	// Print the best chromosome.
	best := geneticAlgorithm.Best()
	fmt.Printf("Best chromosome: %v, Fitness: %v\n", best, best.Fitness())
}
```

### TSP Visualization

The TSP example generates an SVG visualization (`tsp_route.svg`) that includes:

- **Directional arrows** showing the route flow
- **City labels** with names and coordinates
- **Scaled layout** for optimal viewing
- **Total distance** calculation
- **Professional styling** with proper fonts and colors

## Configuration Options

The genetic algorithm supports various configuration options:

- `WithPopulation(population)` - Set initial population
- `WithGenerations(n)` - Number of generations to run
- `WithMutationRate(rate)` - Probability of mutation (0.0 to 1.0)
- `WithCrossoverRate(rate)` - Probability of crossover (0.0 to 1.0)
- `WithElitism(enabled)` - Whether to preserve best chromosome
- `WithSelector(selector)` - Custom selection algorithm

## Implementing Custom Problems

To implement your own optimization problem:

1. **Create a struct** that implements the `ga.Chromosome` interface:
   - `Fitness() float64` - Returns the fitness score
   - `Crossover(other ga.Chromosome) ga.Chromosome` - Creates offspring
   - `Mutate()` - Modifies the chromosome
   - `Clone() ga.Chromosome` - Creates a deep copy

2. **Initialize a population** of your chromosomes

3. **Configure and run** the genetic algorithm

See the provided examples for detailed implementation patterns.

## Testing

Run the test suite:
```bash
make test
# or
go test ./...
```

## Makefile Commands

- `make build` - Build the CLI application
- `make test` - Run tests
- `make fmt` - Format code
- `make example-run` - Run One-Max example
- `make example-run-tsp` - Run TSP example
- `make clean` - Clean build artifacts

## Project Structure
```
├── cmd/ga/           # CLI application
├── ga/               # Core library
│   ├── ga.go         # Main genetic algorithm
│   ├── tsp.go        # TSP implementation
│   ├── visualize.go  # SVG visualization
│   └── ga_test.go    # Tests
├── examples/         # Example data files
├── Makefile          # Build automation
└── README.md         # This file
```

## License

Apache-2.0
```

**Changes made:**
1. Added the `Clone()` method to the OneMaxChromosome example (after `Mutate()`)
2. Updated "Implementing Custom Problems" section to include `Clone()` in the interface requirements

**Commit message:**
```
Docs: Add Clone() method to README example

- Include Clone() implementation in OneMaxChromosome example
- Update interface requirements to include Clone() method
- Ensures README example compiles with current Chromosome interface

Users following the README example would get compile errors without
the Clone() method since it's now required by the interface.