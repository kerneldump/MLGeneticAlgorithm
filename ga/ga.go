package ga

import (
	"math/rand"
	"sort"
)

// Chromosome represents a candidate solution.
type Chromosome interface {
	Fitness() float64
	Crossover(other Chromosome) Chromosome
	Mutate()
}

// Selector is an interface for selection algorithms.
type Selector interface {
	Select(population []Chromosome) []Chromosome
}

// GA is the main struct for the genetic algorithm.
type GA struct {
	Population     []Chromosome
	Generations    int
	MutationRate   float64
	CrossoverRate  float64
	Elitism        bool
	selector       Selector
	BestChromosome Chromosome
}

// New creates a new genetic algorithm.
func New(options ...func(*GA)) *GA {
	ga := &GA{
		Generations:   100,
		MutationRate:  0.01,
		CrossoverRate: 0.8,
		Elitism:       true,
	}
	for _, option := range options {
		option(ga)
	}
	if ga.selector == nil {
		ga.selector = &TournamentSelector{
			TournamentSize: 2,
		}
	}
	return ga
}

// WithPopulation sets the initial population.
func WithPopulation(population []Chromosome) func(*GA) {
	return func(ga *GA) {
		ga.Population = population
	}
}

// WithGenerations sets the number of generations.
func WithGenerations(generations int) func(*GA) {
	return func(ga *GA) {
		ga.Generations = generations
	}
}

// WithMutationRate sets the mutation rate.
func WithMutationRate(mutationRate float64) func(*GA) {
	return func(ga *GA) {
		ga.MutationRate = mutationRate
	}
}

// WithCrossoverRate sets the crossover rate.
func WithCrossoverRate(crossoverRate float64) func(*GA) {
	return func(ga *GA) {
		ga.CrossoverRate = crossoverRate
	}
}

// WithElitism sets the elitism flag.
func WithElitism(elitism bool) func(*GA) {
	return func(ga *GA) {
		ga.Elitism = elitism
	}
}

// WithSelector sets the selection algorithm.
func WithSelector(selector Selector) func(*GA) {
	return func(ga *GA) {
		ga.selector = selector
	}
}

// Run runs the genetic algorithm.
func (ga *GA) Run() {
	if len(ga.Population) == 0 {
		return
	}

	for i := 0; i < ga.Generations; i++ {
		// Sort the population by fitness.
		sort.Slice(ga.Population, func(i, j int) bool {
			return ga.Population[i].Fitness() > ga.Population[j].Fitness()
		})

		// Update the best chromosome.
		if ga.BestChromosome == nil || ga.Population[0].Fitness() > ga.BestChromosome.Fitness() {
			ga.BestChromosome = ga.Population[0]
		}

		// Create the next generation.
		nextGeneration := make([]Chromosome, len(ga.Population))
		nextIndex := 0

		// Apply elitism if enabled
		if ga.Elitism {
			nextGeneration[0] = ga.BestChromosome
			nextIndex = 1
		}

		// Fill the rest of the population
		for nextIndex < len(ga.Population) {
			// Select parents.
			parents := ga.selector.Select(ga.Population)

			var offspring Chromosome
			// Crossover.
			if rand.Float64() < ga.CrossoverRate {
				offspring = parents[0].Crossover(parents[1])
			} else {
				// If no crossover, clone the first parent
				offspring = parents[0].Crossover(parents[0])
			}

			// Mutation.
			if rand.Float64() < ga.MutationRate {
				offspring.Mutate()
			}

			nextGeneration[nextIndex] = offspring
			nextIndex++
		}

		ga.Population = nextGeneration
	}
}

// Best returns the best chromosome found.
func (ga *GA) Best() Chromosome {
	return ga.BestChromosome
}

// TournamentSelector is a selection algorithm that uses tournament selection.
type TournamentSelector struct {
	TournamentSize int
}

// Select selects parents using tournament selection.
func (s *TournamentSelector) Select(population []Chromosome) []Chromosome {
	if len(population) == 0 {
		return []Chromosome{}
	}

	// Ensure tournament size is valid
	tournamentSize := s.TournamentSize
	if tournamentSize <= 0 {
		tournamentSize = 2
	}
	if tournamentSize > len(population) {
		tournamentSize = len(population)
	}

	parents := make([]Chromosome, 2)
	for i := 0; i < 2; i++ {
		// Run tournament
		best := population[rand.Intn(len(population))]
		for j := 1; j < tournamentSize; j++ {
			competitor := population[rand.Intn(len(population))]
			if competitor.Fitness() > best.Fitness() {
				best = competitor
			}
		}
		parents[i] = best
	}
	return parents
}
