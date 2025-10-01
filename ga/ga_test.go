package ga

import (
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

func TestTournamentSelector(t *testing.T) {
	population := []Chromosome{
		&MockChromosome{fitness: 1.0},
		&MockChromosome{fitness: 2.0},
		&MockChromosome{fitness: 3.0},
		&MockChromosome{fitness: 4.0},
		&MockChromosome{fitness: 5.0},
	}

	selector := &TournamentSelector{TournamentSize: 2}
	parents := selector.Select(population)

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

	err := ga.Run()  // Now Run() returns error
	if err == nil {
		t.Error("Expected error when running with invalid configuration, got nil")
	}
}