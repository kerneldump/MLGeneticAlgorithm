package ga

import (
	"math"
	"math/rand"
)

// City represents a city in the TSP problem.
type City struct {
	Name string
	X    float64
	Y    float64
}

// TSPChromosome is a chromosome for the TSP problem.
type TSPChromosome struct {
	Route []City
}

// Fitness calculates the fitness of the chromosome (total distance of the route).
func (c *TSPChromosome) Fitness() float64 {
	if len(c.Route) < 2 {
		return 0
	}

	var totalDistance float64
	for i := 0; i < len(c.Route)-1; i++ {
		totalDistance += distance(c.Route[i], c.Route[i+1])
	}
	totalDistance += distance(c.Route[len(c.Route)-1], c.Route[0]) // Return to start

	if totalDistance == 0 {
		return 0
	}
	return 1 / totalDistance // We want to minimize distance, so we maximize 1/distance
}

// Crossover creates a new chromosome using Order Crossover (OX1).
func (c *TSPChromosome) Crossover(other Chromosome) Chromosome {
	parent1 := c.Route
	parent2 := other.(*TSPChromosome).Route

	if len(parent1) != len(parent2) || len(parent1) < 2 {
		// If parents are incompatible, return a copy of parent1
		childRoute := make([]City, len(parent1))
		copy(childRoute, parent1)
		return &TSPChromosome{Route: childRoute}
	}

	// Order Crossover (OX1)
	start := rand.Intn(len(parent1))
	end := rand.Intn(len(parent1))

	if start > end {
		start, end = end, start
	}

	childRoute := make([]City, len(parent1))

	// Copy the selected segment from parent1
	for i := start; i <= end; i++ {
		childRoute[i] = parent1[i]
	}

	// Create a set for O(1) lookup of cities already in child
	// Use city names for comparison to avoid floating-point equality issues
	inChild := make(map[string]bool)
	for i := start; i <= end; i++ {
		inChild[parent1[i].Name] = true
	}

	// Fill remaining positions with cities from parent2 in order
	childIndex := (end + 1) % len(parent1)
	for i := 0; i < len(parent2); i++ {
		parent2Index := (end + 1 + i) % len(parent2)
		city := parent2[parent2Index]

		if !inChild[city.Name] {
			childRoute[childIndex] = city
			childIndex = (childIndex + 1) % len(parent1)
		}
	}

	return &TSPChromosome{Route: childRoute}
}

// Mutate randomly swaps two cities in the route.
func (c *TSPChromosome) Mutate() {
	if len(c.Route) < 2 {
		return
	}

	i := rand.Intn(len(c.Route))
	j := rand.Intn(len(c.Route))

	// Ensure we're swapping different cities
	for i == j {
		j = rand.Intn(len(c.Route))
	}

	c.Route[i], c.Route[j] = c.Route[j], c.Route[i]
}

func distance(city1, city2 City) float64 {
	dx := city1.X - city2.X
	dy := city1.Y - city2.Y
	return math.Sqrt(dx*dx + dy*dy)
}
