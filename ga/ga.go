// Package ga provides a flexible and extensible genetic algorithm framework.
//
// A genetic algorithm is an optimization technique inspired by natural selection.
// It works by evolving a population of candidate solutions over multiple generations,
// using operations like selection, crossover, and mutation.
//
// Basic usage:
//
//	population := []ga.Chromosome{...}
//	algorithm := ga.New(
//	    ga.WithPopulation(population),
//	    ga.WithGenerations(100),
//	    ga.WithMutationRate(0.01),
//	)
//	err := algorithm.Run()
//	best := algorithm.Best()
package ga

import (
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"time"
)

// Chromosome represents a candidate solution in the genetic algorithm.
// Implementations must define how to evaluate fitness, combine with other
// chromosomes (crossover), and introduce random changes (mutation).
type Chromosome interface {
	// Fitness returns the quality of this solution. Higher values are better.
	Fitness() float64

	// Crossover combines this chromosome with another to create a new offspring.
	Crossover(other Chromosome) Chromosome

	// Mutate introduces a random change to this chromosome.
	Mutate()
}

// Selector defines how parent chromosomes are chosen for reproduction.
// Different selection strategies (tournament, roulette, rank-based) can be
// implemented by satisfying this interface.
//
// THREAD SAFETY NOTE: The rng parameter must be used for all random operations
// instead of the global math/rand to ensure thread-safe concurrent execution.
type Selector interface {
	// Select chooses parent chromosomes from the population for breeding.
	// The rng parameter provides a thread-safe random number generator that
	// must be used for all random operations within the selector.
	// Returns a slice of selected parents (typically 2).
	Select(population []Chromosome, rng *rand.Rand) []Chromosome
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

// New creates a new genetic algorithm with default settings.
// Use the With* option functions to customize the algorithm.
//
// Default settings:
//   - 100 generations
//   - 0.01 mutation rate
//   - 0.8 crossover rate
//   - Elitism enabled
//   - Tournament selection (size 2)
//   - Random seed from current time
//
// Example:
//
//	ga := ga.New(
//	    ga.WithPopulation(population),
//	    ga.WithGenerations(200),
//	    ga.WithMutationRate(0.02),
//	)
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

// Validate checks if the GA configuration is valid and returns an error
// if any issues are found.
//
// Checks include:
//   - Population is not nil or empty
//   - Generations is at least 1
//   - MutationRate is between 0 and 1
//   - CrossoverRate is between 0 and 1
//   - Selector is not nil
//   - No nil chromosomes in population
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

// WithPopulation sets the initial population of chromosomes.
// This is typically required as there is no default population.
func WithPopulation(population []Chromosome) func(*GA) {
	return func(ga *GA) {
		ga.Population = population
	}
}

// WithGenerations sets the maximum number of generations to evolve.
// The algorithm may stop earlier if convergence is detected.
func WithGenerations(generations int) func(*GA) {
	return func(ga *GA) {
		ga.Generations = generations
	}
}

// WithMutationRate sets the probability of mutation occurring (0.0 to 1.0).
// Higher values introduce more randomness and exploration.
// Typical values are between 0.01 and 0.1.
func WithMutationRate(mutationRate float64) func(*GA) {
	return func(ga *GA) {
		ga.MutationRate = mutationRate
	}
}

// WithCrossoverRate sets the probability of crossover occurring (0.0 to 1.0).
// Higher values favor combining parent genes more often.
// Typical values are between 0.7 and 0.95.
func WithCrossoverRate(crossoverRate float64) func(*GA) {
	return func(ga *GA) {
		ga.CrossoverRate = crossoverRate
	}
}

// WithElitism determines whether the best chromosome is always preserved.
// When enabled, the best solution is guaranteed to survive to the next generation.
// This prevents losing good solutions but may slow convergence.
func WithElitism(elitism bool) func(*GA) {
	return func(ga *GA) {
		ga.Elitism = elitism
	}
}

// WithSelector sets a custom selection algorithm.
// If not specified, tournament selection with size 2 is used by default.
//
// IMPORTANT: Custom selectors must use the provided rng parameter for all
// random operations to ensure thread safety.
func WithSelector(selector Selector) func(*GA) {
	return func(ga *GA) {
		ga.selector = selector
	}
}

// WithProgressCallback sets a callback function to monitor the algorithm's progress.
// The callback is invoked after each generation with the generation number and
// current best chromosome.
//
// Example:
//
//	ga.WithProgressCallback(func(gen int, best ga.Chromosome) {
//	    fmt.Printf("Generation %d: fitness = %.4f\n", gen, best.Fitness())
//	})
func WithProgressCallback(callback func(generation int, best Chromosome)) func(*GA) {
	return func(ga *GA) {
		ga.progressCallback = callback
	}
}

// WithRandomSeed sets a specific seed for the random number generator.
// This is useful for reproducible results in testing and debugging.
// If not specified, a time-based seed is used automatically.
//
// Example:
//
//	ga.WithRandomSeed(12345)  // Same seed produces same results
func WithRandomSeed(seed int64) func(*GA) {
	return func(ga *GA) {
		ga.rng = rand.New(rand.NewSource(seed))
	}
}

// WithConvergence enables early stopping when the algorithm plateaus.
// The algorithm stops if the best fitness doesn't improve by at least 'threshold'
// for 'generations' consecutive generations.
//
// Parameters:
//   - generations: Number of generations without improvement before stopping
//   - threshold: Minimum fitness improvement to count as progress (use 0 for any improvement)
//
// Example:
//
//	ga.WithConvergence(20, 0.0001)  // Stop if no improvement > 0.0001 for 20 generations
func WithConvergence(generations int, threshold float64) func(*GA) {
	return func(ga *GA) {
		ga.convergenceGenerations = generations
		ga.convergenceThreshold = threshold
	}
}

// Run executes the genetic algorithm for the configured number of generations
// or until convergence is detected.
//
// The algorithm:
//  1. Validates the configuration
//  2. For each generation:
//     - Sorts population by fitness
//     - Updates best chromosome
//     - Checks for convergence (if enabled)
//     - Calls progress callback (if provided)
//     - Creates next generation via selection, crossover, and mutation
//  3. Returns nil on success, or an error if configuration is invalid
//
// THREAD SAFETY: Each GA instance has its own RNG and can run concurrently
// with other GA instances. However, do not call Run() on the same GA instance
// from multiple goroutines simultaneously.
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
			// THREAD SAFETY FIX: Lock for entire selection and reproduction
			// Pass ga.rng to selector instead of relying on global rand
			ga.mu.Lock()

			// Select parents using thread-safe RNG
			parents := ga.selector.Select(ga.Population, ga.rng)

			var offspring Chromosome
			// Crossover.
			shouldCrossover := ga.rng.Float64() < ga.CrossoverRate

			if shouldCrossover {
				offspring = parents[0].Crossover(parents[1])
			} else {
				// If no crossover, clone the first parent
				offspring = parents[0].Crossover(parents[0])
			}

			// Mutation.
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

// Best returns the best chromosome found during the algorithm's execution.
// Returns nil if Run() has not been called yet.
func (ga *GA) Best() Chromosome {
	return ga.BestChromosome
}

// TournamentSelector implements tournament selection for parent selection.
// It randomly selects TournamentSize individuals and chooses the fittest one.
// This process is repeated to select multiple parents.
//
// Tournament selection balances selection pressure with diversity:
//   - Larger tournament sizes increase selection pressure (favor fit individuals)
//   - Smaller tournament sizes maintain more diversity
type TournamentSelector struct {
	// TournamentSize is the number of individuals competing in each tournament.
	// Typical values are 2-5. Default is 2 if not specified or if <= 0.
	TournamentSize int
}

// Select chooses parent chromosomes from the population using tournament selection.
// Returns a slice of 2 parents for breeding.
//
// The selection process:
//  1. Randomly select TournamentSize individuals
//  2. Choose the one with highest fitness
//  3. Repeat to select the second parent
//
// THREAD SAFETY: Uses the provided rng parameter instead of global math/rand,
// ensuring safe concurrent execution across multiple GA instances.
func (s *TournamentSelector) Select(population []Chromosome, rng *rand.Rand) []Chromosome {
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
		// Run tournament using provided thread-safe RNG
		best := population[rng.Intn(len(population))]
		for j := 1; j < tournamentSize; j++ {
			competitor := population[rng.Intn(len(population))]
			if competitor.Fitness() > best.Fitness() {
				best = competitor
			}
		}
		parents[i] = best
	}
	return parents
}
