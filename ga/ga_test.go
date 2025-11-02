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

func (c *MockChromosome) Clone() Chromosome {
	return &MockChromosome{fitness: c.fitness}
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

// TestValidatePopulationSizeLimit tests the population size validation
func TestValidatePopulationSizeLimit(t *testing.T) {
	// Create population exceeding the limit
	largePopulation := make([]Chromosome, 150000)
	for i := range largePopulation {
		largePopulation[i] = &MockChromosome{fitness: 1.0}
	}

	ga := New(
		WithPopulation(largePopulation),
		WithGenerations(10),
	)

	err := ga.Validate()
	if err == nil {
		t.Error("Expected error for oversized population (>100000), got nil")
	}

	// Verify error message mentions the limit
	expectedMsg := "exceeds recommended maximum"
	if err != nil && len(err.Error()) > 0 {
		if !contains(err.Error(), expectedMsg) {
			t.Errorf("Expected error message to contain '%s', got: %v", expectedMsg, err)
		}
	}
}

// TestValidateNegativeGenerations tests negative generation values
func TestValidateNegativeGenerations(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(-5),
	)

	err := ga.Validate()
	if err == nil {
		t.Error("Expected error for negative generations, got nil")
	}
}

// TestConvergenceWithNegativeFitness tests convergence detection with negative fitness values
func TestConvergenceWithNegativeFitness(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: -10.0},
		&MockChromosome{fitness: -5.0},
		&MockChromosome{fitness: -8.0},
	}

	callbackCount := 0
	ga := New(
		WithPopulation(population),
		WithGenerations(100),
		WithConvergence(3, 0.01), // Stop after 3 gens without improvement > 0.01
		WithRandomSeed(42),
		WithProgressCallback(func(generation int, best Chromosome) {
			callbackCount++
		}),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Should converge early (not run all 100 generations)
	if callbackCount >= 100 {
		t.Errorf("Expected early convergence, but ran all %d generations", callbackCount)
	}

	// Best fitness should be the least negative (closest to 0)
	bestFitness := ga.Best().Fitness()
	if bestFitness < -5.0 {
		t.Errorf("Expected best fitness around -5.0, got %f", bestFitness)
	}
}

// TestConvergenceWithZeroThreshold tests convergence with zero threshold
func TestConvergenceWithZeroThreshold(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	callbackCount := 0
	ga := New(
		WithPopulation(population),
		WithGenerations(50),
		WithConvergence(5, 0.0), // Stop after 5 gens with ANY improvement threshold
		WithRandomSeed(42),
		WithProgressCallback(func(generation int, best Chromosome) {
			callbackCount++
		}),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// With zero threshold, should detect convergence relatively quickly
	if callbackCount > 30 {
		t.Logf("Convergence with zero threshold took %d generations (expected < 30)", callbackCount)
	}
}

// TestEmptySelector tests behavior with empty population in selector
func TestEmptySelectorPopulation(t *testing.T) {
	selector := &TournamentSelector{TournamentSize: 2}
	rng := rand.New(rand.NewSource(12345))

	parents := selector.Select([]Chromosome{}, rng)

	if len(parents) != 0 {
		t.Errorf("Expected empty parents slice for empty population, got %d parents", len(parents))
	}
}

// TestTournamentSelectorInvalidSize tests tournament selector with invalid sizes
func TestTournamentSelectorInvalidSize(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	tests := []struct {
		name           string
		tournamentSize int
		expectParents  int
	}{
		{"zero size", 0, 2},             // Should default to 2
		{"negative size", -5, 2},        // Should default to 2
		{"size larger than pop", 10, 2}, // Should clamp to population size
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			selector := &TournamentSelector{TournamentSize: tt.tournamentSize}
			rng := rand.New(rand.NewSource(12345))

			parents := selector.Select(population, rng)

			if len(parents) != tt.expectParents {
				t.Errorf("Expected %d parents, got %d", tt.expectParents, len(parents))
			}
		})
	}
}

// TestElitismPreservation verifies elitism keeps the best chromosome
func TestElitismPreservation(t *testing.T) {
	// Create population where best has unique fitness
	population := []Chromosome{
		&MockChromosome{fitness: 100.0}, // Best
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(10),
		WithElitism(true),
		WithRandomSeed(42),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Best should still have fitness >= 100 (might increase due to mutations)
	bestFitness := ga.Best().Fitness()
	if bestFitness < 100.0 {
		t.Errorf("Elitism failed: best fitness %f is less than initial best 100.0", bestFitness)
	}
}

// TestNoElitismCanLoseBest verifies that without elitism, best can be lost
func TestNoElitismBehavior(t *testing.T) {
	// Create population where best is very different
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 100.0}, // Best, but only 1 copy
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(5),
		WithElitism(false), // No elitism
		WithMutationRate(0.5),
		WithRandomSeed(42),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Just verify it runs without error - best might or might not be preserved
	// This test mainly verifies the non-elitism code path works
	if ga.Best() == nil {
		t.Error("Expected best chromosome to be set even without elitism")
	}
}

// TestSingleChromosomePopulation tests GA with only one chromosome
func TestSingleChromosomePopulation(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 5.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(10),
		WithRandomSeed(42),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run with single chromosome failed: %v", err)
	}

	if ga.Best() == nil {
		t.Error("Expected best chromosome to be set")
	}
}

// TestZeroMutationRate tests GA with no mutations
func TestZeroMutationRate(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(10),
		WithMutationRate(0.0), // No mutations
		WithRandomSeed(42),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run with zero mutation rate failed: %v", err)
	}

	if ga.Best() == nil {
		t.Error("Expected best chromosome to be set")
	}
}

// TestZeroCrossoverRate tests GA with no crossover
func TestZeroCrossoverRate(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
	}

	ga := New(
		WithPopulation(population),
		WithGenerations(10),
		WithCrossoverRate(0.0), // No crossover - only cloning
		WithRandomSeed(42),
	)

	err := ga.Run()
	if err != nil {
		t.Fatalf("Run with zero crossover rate failed: %v", err)
	}

	if ga.Best() == nil {
		t.Error("Expected best chromosome to be set")
	}
}

// Helper function to check if string contains substring
func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > len(substr) && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
