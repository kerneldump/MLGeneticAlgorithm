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
