package ga

import (
	"math/rand"
	"sync"
	"testing"
)

type MockChromosome struct {
	fitness float64
}

func (c *MockChromosome) Fitness() float64 {
	return c.fitness
}

func (c *MockChromosome) Crossover(other Chromosome) Chromosome {
	return &MockChromosome{fitness: (c.fitness + other.Fitness()) / 2}
}

func (c *MockChromosome) Mutate() {
	c.fitness += 0.1
}

// NEW TEST: Verify TournamentSelector uses provided RNG
func TestTournamentSelectorUsesProvidedRNG(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
		&MockChromosome{fitness: 4.0},
		&MockChromosome{fitness: 5.0},
	}

	selector := &TournamentSelector{TournamentSize: 2}

	// Create two RNGs with same seed
	rng1 := rand.New(rand.NewSource(12345))
	rng2 := rand.New(rand.NewSource(12345))

	// Selections should be identical with same RNG seed
	parents1 := selector.Select(population, rng1)
	parents2 := selector.Select(population, rng2)

	if len(parents1) != 2 || len(parents2) != 2 {
		t.Errorf("Expected 2 parents, got %d and %d", len(parents1), len(parents2))
	}

	// With same seed, should get same parents
	if parents1[0].Fitness() != parents2[0].Fitness() || parents1[1].Fitness() != parents2[1].Fitness() {
		t.Errorf("Expected identical parents with same RNG seed")
	}
}

// NEW TEST: Verify concurrent execution safety
func TestConcurrentGAExecution(t *testing.T) {
	// Create multiple GA instances
	numGoroutines := 10
	var wg sync.WaitGroup
	errors := make(chan error, numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(seed int64) {
			defer wg.Done()

			// Each goroutine gets its own population
			population := []Chromosome{
				&MockChromosome{fitness: 1.0},
				&MockChromosome{fitness: 2.0},
				&MockChromosome{fitness: 3.0},
			}

			ga := New(
				WithPopulation(population),
				WithGenerations(20),
				WithRandomSeed(seed),
			)

			if err := ga.Run(); err != nil {
				errors <- err
			}
		}(int64(i))
	}

	wg.Wait()
	close(errors)

	// Check for any errors
	for err := range errors {
		t.Errorf("Concurrent execution failed: %v", err)
	}
}

// NEW TEST: Verify RNG determinism
func TestDeterministicWithSeed(t *testing.T) {
	createGA := func(seed int64) *GA {
		population := []Chromosome{
			&MockChromosome{fitness: 1.0},
			&MockChromosome{fitness: 2.0},
			&MockChromosome{fitness: 3.0},
		}
		return New(
			WithPopulation(population),
			WithGenerations(10),
			WithRandomSeed(seed),
		)
	}

	ga1 := createGA(42)
	ga2 := createGA(42)

	if err := ga1.Run(); err != nil {
		t.Fatalf("GA1 failed: %v", err)
	}
	if err := ga2.Run(); err != nil {
		t.Fatalf("GA2 failed: %v", err)
	}

	// With same seed, results should be identical
	if ga1.Best().Fitness() != ga2.Best().Fitness() {
		t.Errorf("Expected identical results with same seed, got %f and %f",
			ga1.Best().Fitness(), ga2.Best().Fitness())
	}
}

// Existing tests updated to work with new Selector interface

func TestTournamentSelector(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
		&MockChromosome{fitness: 4.0},
		&MockChromosome{fitness: 5.0},
	}

	selector := &TournamentSelector{TournamentSize: 2}
	rng := rand.New(rand.NewSource(12345))
	parents := selector.Select(population, rng)

	if len(parents) != 2 {
		t.Errorf("Expected 2 parents, but got %d", len(parents))
	}
}

func TestValidateEmptyPopulation(t *testing.T) {
	ga := New(
		WithPopulation([]Chromosome{}),
		WithGenerations(10),
	)

	err := ga.Validate()
	if err == nil {
		t.Error("Expected error for empty population, got nil")
	}
}

func TestValidateNilPopulation(t *testing.T) {
	ga := New(
		WithGenerations(10),
	)
	ga.Population = nil

	err := ga.Validate()
	if err == nil {
		t.Error("Expected error for nil population, got nil")
	}
}

func TestValidateInvalidGenerations(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(0),
	)

	err := ga.Validate()
	if err == nil {
		t.Error("Expected error for 0 generations, got nil")
	}
}

func TestValidateInvalidMutationRate(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
	}

	tests := []struct {
		name         string
		mutationRate float64
	}{
		{"negative rate", -0.1},
		{"rate too high", 1.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ga := New(
				WithPopulation(population),
				WithMutationRate(tt.mutationRate),
			)

			err := ga.Validate()
			if err == nil {
				t.Errorf("Expected error for mutation rate %f, got nil", tt.mutationRate)
			}
		})
	}
}

func TestValidateInvalidCrossoverRate(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
	}

	tests := []struct {
		name          string
		crossoverRate float64
	}{
		{"negative rate", -0.1},
		{"rate too high", 1.5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ga := New(
				WithPopulation(population),
				WithCrossoverRate(tt.crossoverRate),
			)

			err := ga.Validate()
			if err == nil {
				t.Errorf("Expected error for crossover rate %f, got nil", tt.crossoverRate)
			}
		})
	}
}

func TestValidateNilChromosomeInPopulation(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		nil,
		&MockChromosome{fitness: 3.0},
	}

	ga := New(
		WithPopulation(population),
	)

	err := ga.Validate()
	if err == nil {
		t.Error("Expected error for nil chromosome in population, got nil")
	}
}

func TestValidateValidConfiguration(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(10),
		WithMutationRate(0.01),
		WithCrossoverRate(0.8),
	)

	err := ga.Validate()
	if err != nil {
		t.Errorf("Expected no error for valid configuration, got: %v", err)
	}
}

func TestRunWithInvalidConfiguration(t *testing.T) {
	ga := New(
		WithPopulation([]Chromosome{}),
	)

	err := ga.Run()
	if err == nil {
		t.Error("Expected error when running with invalid configuration, got nil")
	}
}

func TestProgressCallback(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	callbackCalled := false
	var lastGeneration int
	var lastBest Chromosome

	ga := New(
		WithPopulation(population),
		WithGenerations(5),
		WithProgressCallback(func(generation int, best Chromosome) {
			callbackCalled = true
			lastGeneration = generation
			lastBest = best
		}),
	)

	err := ga.Run()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	if !callbackCalled {
		t.Error("Progress callback was never called")
	}

	if lastGeneration != 4 {
		t.Errorf("Expected last generation to be 4, got %d", lastGeneration)
	}

	if lastBest == nil {
		t.Error("Expected best chromosome to be set")
	}
}

func TestReproducibleWithSeed(t *testing.T) {
	population1 := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	population2 := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	ga1 := New(
		WithPopulation(population1),
		WithGenerations(10),
		WithRandomSeed(12345),
	)

	ga2 := New(
		WithPopulation(population2),
		WithGenerations(10),
		WithRandomSeed(12345),
	)

	err1 := ga1.Run()
	err2 := ga2.Run()

	if err1 != nil || err2 != nil {
		t.Errorf("Unexpected errors: %v, %v", err1, err2)
	}

	if ga1.Best().Fitness() != ga2.Best().Fitness() {
		t.Errorf("Expected identical results with same seed, got %f and %f",
			ga1.Best().Fitness(), ga2.Best().Fitness())
	}
}
