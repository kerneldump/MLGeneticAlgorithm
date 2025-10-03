package ga

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"
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
	Population             []Chromosome
	Generations            int
	MutationRate           float64
	CrossoverRate          float64
	Elitism                bool
	selector               Selector
	BestChromosome         Chromosome
	progressCallback       func(generation int, best Chromosome)
	rng                    *rand.Rand
	mu                     sync.Mutex
	convergenceGenerations int
	convergenceThreshold   float64
}

// New creates a new genetic algorithm.
func New(options ...func(*GA)) *GA {
	ga := &GA{
		Generations:   100,
		MutationRate:  0.01,
		CrossoverRate: 0.8,
		Elitism:       true,
		rng:           rand.New(rand.NewSource(time.Now().UnixNano())),
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

// Validate checks if the GA configuration is valid.
func (ga *GA) Validate() error {
	if len(ga.Population) == 0 {
		return fmt.Errorf("population cannot be nil or empty")
	}

	if ga.Generations < 1 {
		return fmt.Errorf("generations must be at least 1, got %d", ga.Generations)
	}

	if ga.MutationRate < 0 || ga.MutationRate > 1 {
		return fmt.Errorf("mutation rate must be between 0 and 1, got %f", ga.MutationRate)
	}

	if ga.CrossoverRate < 0 || ga.CrossoverRate > 1 {
		return fmt.Errorf("crossover rate must be between 0 and 1, got %f", ga.CrossoverRate)
	}

	if ga.selector == nil {
		return fmt.Errorf("selector cannot be nil")
	}

	// Validate population contains no nil chromosomes
	for i, chromosome := range ga.Population {
		if chromosome == nil {
			return fmt.Errorf("population contains nil chromosome at index %d", i)
		}
	}

	return nil
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

// WithProgressCallback sets a callback function to monitor progress.
func WithProgressCallback(callback func(generation int, best Chromosome)) func(*GA) {
	return func(ga *GA) {
		ga.progressCallback = callback
	}
}

// WithRandomSeed sets a specific seed for the random number generator.
// Useful for reproducible results in testing.
func WithRandomSeed(seed int64) func(*GA) {
	return func(ga *GA) {
		ga.rng = rand.New(rand.NewSource(seed))
	}
}

// WithConvergence enables early stopping if fitness doesn't improve.
// generations: number of generations without improvement before stopping
// threshold: minimum fitness improvement to be considered progress (use 0 for any improvement)
func WithConvergence(generations int, threshold float64) func(*GA) {
	return func(ga *GA) {
		ga.convergenceGenerations = generations
		ga.convergenceThreshold = threshold
	}
}

// Run runs the genetic algorithm.
func (ga *GA) Run() error {
	// Validate configuration before running
	if err := ga.Validate(); err != nil {
		return fmt.Errorf("invalid GA configuration: %w", err)
	}

	var lastBestFitness float64
	var generationsWithoutImprovement int

	for i := 0; i < ga.Generations; i++ {
		// Sort the population by fitness.
		sort.Slice(ga.Population, func(i, j int) bool {
			return ga.Population[i].Fitness() > ga.Population[j].Fitness()
		})

		// Update the best chromosome.
		currentBestFitness := ga.Population[0].Fitness()
		if ga.BestChromosome == nil || currentBestFitness > ga.BestChromosome.Fitness() {
			ga.BestChromosome = ga.Population[0]
		}

		// Check for convergence
		if ga.convergenceGenerations > 0 {
			improvement := currentBestFitness - lastBestFitness
			if improvement > ga.convergenceThreshold {
				// Significant improvement, reset counter
				generationsWithoutImprovement = 0
			} else {
				// No improvement, increment counter
				generationsWithoutImprovement++
				if generationsWithoutImprovement >= ga.convergenceGenerations {
					// Converged - call callback one last time and exit
					if ga.progressCallback != nil {
						ga.progressCallback(i, ga.BestChromosome)
					}
					return nil
				}
			}
			lastBestFitness = currentBestFitness
		}

		// Call progress callback if provided
		if ga.progressCallback != nil {
			ga.progressCallback(i, ga.BestChromosome)
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
			ga.mu.Lock()
			shouldCrossover := ga.rng.Float64() < ga.CrossoverRate
			ga.mu.Unlock()

			if shouldCrossover {
				offspring = parents[0].Crossover(parents[1])
			} else {
				// If no crossover, clone the first parent
				offspring = parents[0].Crossover(parents[0])
			}

			// Mutation.
			ga.mu.Lock()
			shouldMutate := ga.rng.Float64() < ga.MutationRate
			ga.mu.Unlock()

			if shouldMutate {
				offspring.Mutate()
			}

			nextGeneration[nextIndex] = offspring
			nextIndex++
		}

		ga.Population = nextGeneration
	}

	return nil
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
		// Note: This uses math/rand for backward compatibility with user chromosomes
		// In production, chromosomes should also use the GA's RNG
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
