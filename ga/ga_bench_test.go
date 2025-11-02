package ga

import (
	"fmt"
	"math/rand"
	"testing"
)

// ==================== Selector Benchmarks ====================

// BenchmarkTournamentSelection benchmarks the tournament selection algorithm
func BenchmarkTournamentSelection(b *testing.B) {
	sizes := []int{10, 100, 1000, 10000}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("pop_%d", size), func(b *testing.B) {
			population := make([]Chromosome, size)
			for i := range population {
				population[i] = &MockChromosome{fitness: float64(i)}
			}

			selector := &TournamentSelector{TournamentSize: 5}
			rng := rand.New(rand.NewSource(12345))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				selector.Select(population, rng)
			}
		})
	}
}

// BenchmarkTournamentSelectionSizes benchmarks different tournament sizes
func BenchmarkTournamentSelectionSizes(b *testing.B) {
	population := make([]Chromosome, 1000)
	for i := range population {
		population[i] = &MockChromosome{fitness: float64(i)}
	}

	tournamentSizes := []int{2, 5, 10, 20, 50}

	for _, tournamentSize := range tournamentSizes {
		b.Run(fmt.Sprintf("size_%d", tournamentSize), func(b *testing.B) {
			selector := &TournamentSelector{TournamentSize: tournamentSize}
			rng := rand.New(rand.NewSource(12345))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				selector.Select(population, rng)
			}
		})
	}
}

// ==================== TSP Benchmarks ====================

// BenchmarkTSPCrossover benchmarks TSP Order Crossover (OX1)
func BenchmarkTSPCrossover(b *testing.B) {
	cityCounts := []int{10, 50, 100, 500}

	for _, count := range cityCounts {
		b.Run(fmt.Sprintf("cities_%d", count), func(b *testing.B) {
			cities := make([]City, count)
			for i := range cities {
				cities[i] = City{
					Name: fmt.Sprintf("City%d", i),
					X:    float64(i * 10),
					Y:    float64(i * 15),
				}
			}

			parent1 := &TSPChromosome{Route: cities}
			parent2 := &TSPChromosome{Route: cities}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = parent1.Crossover(parent2)
			}
		})
	}
}

// BenchmarkTSPMutation benchmarks TSP swap mutation
func BenchmarkTSPMutation(b *testing.B) {
	cityCounts := []int{10, 50, 100, 500}

	for _, count := range cityCounts {
		b.Run(fmt.Sprintf("cities_%d", count), func(b *testing.B) {
			cities := make([]City, count)
			for i := range cities {
				cities[i] = City{
					Name: fmt.Sprintf("City%d", i),
					X:    float64(i * 10),
					Y:    float64(i * 15),
				}
			}

			chromosome := &TSPChromosome{Route: cities}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				chromosome.Mutate()
			}
		})
	}
}

// BenchmarkTSPFitness benchmarks TSP fitness calculation
func BenchmarkTSPFitness(b *testing.B) {
	cityCounts := []int{10, 50, 100, 500, 1000}

	for _, count := range cityCounts {
		b.Run(fmt.Sprintf("cities_%d", count), func(b *testing.B) {
			cities := make([]City, count)
			for i := range cities {
				cities[i] = City{
					Name: fmt.Sprintf("City%d", i),
					X:    float64(i * 10),
					Y:    float64(i * 15),
				}
			}

			chromosome := &TSPChromosome{Route: cities}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = chromosome.Fitness()
			}
		})
	}
}

// BenchmarkTSPClone benchmarks TSP chromosome cloning
func BenchmarkTSPClone(b *testing.B) {
	cityCounts := []int{10, 50, 100, 500}

	for _, count := range cityCounts {
		b.Run(fmt.Sprintf("cities_%d", count), func(b *testing.B) {
			cities := make([]City, count)
			for i := range cities {
				cities[i] = City{
					Name: fmt.Sprintf("City%d", i),
					X:    float64(i * 10),
					Y:    float64(i * 15),
				}
			}

			chromosome := &TSPChromosome{Route: cities}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = chromosome.Clone()
			}
		})
	}
}

// ==================== GA Full Run Benchmarks ====================

// BenchmarkGARun benchmarks a complete GA execution
func BenchmarkGARun(b *testing.B) {
	configs := []struct {
		name        string
		popSize     int
		generations int
	}{
		{"small_10x10", 10, 10},
		{"medium_100x50", 100, 50},
		{"large_500x100", 500, 100},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				population := make([]Chromosome, config.popSize)
				for j := range population {
					population[j] = &MockChromosome{fitness: float64(j)}
				}
				b.StartTimer()

				ga := New(
					WithPopulation(population),
					WithGenerations(config.generations),
					WithRandomSeed(12345),
				)
				_ = ga.Run()
			}
		})
	}
}

// BenchmarkGARunTSP benchmarks GA with TSP problem
func BenchmarkGARunTSP(b *testing.B) {
	configs := []struct {
		name        string
		cities      int
		popSize     int
		generations int
	}{
		{"tsp_10cities_50pop_20gen", 10, 50, 20},
		{"tsp_20cities_100pop_50gen", 20, 100, 50},
		{"tsp_50cities_100pop_100gen", 50, 100, 100},
	}

	for _, config := range configs {
		b.Run(config.name, func(b *testing.B) {
			// Create cities once outside the timing loop
			cities := make([]City, config.cities)
			for i := range cities {
				cities[i] = City{
					Name: fmt.Sprintf("City%d", i),
					X:    float64(i * 10),
					Y:    float64(i * 15),
				}
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				population := make([]Chromosome, config.popSize)
				for j := range population {
					route := make([]City, len(cities))
					copy(route, cities)
					population[j] = &TSPChromosome{Route: route}
				}
				b.StartTimer()

				ga := New(
					WithPopulation(population),
					WithGenerations(config.generations),
					WithMutationRate(0.02),
					WithCrossoverRate(0.85),
					WithRandomSeed(12345),
				)
				_ = ga.Run()
			}
		})
	}
}

// ==================== Configuration Benchmarks ====================

// BenchmarkElitismVsNoElitism compares performance with and without elitism
func BenchmarkElitismVsNoElitism(b *testing.B) {
	population := make([]Chromosome, 100)
	for i := range population {
		population[i] = &MockChromosome{fitness: float64(i)}
	}

	b.Run("with_elitism", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			pop := make([]Chromosome, len(population))
			copy(pop, population)
			b.StartTimer()

			ga := New(
				WithPopulation(pop),
				WithGenerations(50),
				WithElitism(true),
				WithRandomSeed(12345),
			)
			_ = ga.Run()
		}
	})

	b.Run("without_elitism", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			b.StopTimer()
			pop := make([]Chromosome, len(population))
			copy(pop, population)
			b.StartTimer()

			ga := New(
				WithPopulation(pop),
				WithGenerations(50),
				WithElitism(false),
				WithRandomSeed(12345),
			)
			_ = ga.Run()
		}
	})
}

// BenchmarkMutationRates compares different mutation rates
func BenchmarkMutationRates(b *testing.B) {
	mutationRates := []float64{0.0, 0.01, 0.05, 0.1, 0.5}

	for _, rate := range mutationRates {
		b.Run(fmt.Sprintf("rate_%.2f", rate), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				population := make([]Chromosome, 100)
				for j := range population {
					population[j] = &MockChromosome{fitness: float64(j)}
				}
				b.StartTimer()

				ga := New(
					WithPopulation(population),
					WithGenerations(20),
					WithMutationRate(rate),
					WithRandomSeed(12345),
				)
				_ = ga.Run()
			}
		})
	}
}

// BenchmarkCrossoverRates compares different crossover rates
func BenchmarkCrossoverRates(b *testing.B) {
	crossoverRates := []float64{0.0, 0.5, 0.8, 0.95, 1.0}

	for _, rate := range crossoverRates {
		b.Run(fmt.Sprintf("rate_%.2f", rate), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				population := make([]Chromosome, 100)
				for j := range population {
					population[j] = &MockChromosome{fitness: float64(j)}
				}
				b.StartTimer()

				ga := New(
					WithPopulation(population),
					WithGenerations(20),
					WithCrossoverRate(rate),
					WithRandomSeed(12345),
				)
				_ = ga.Run()
			}
		})
	}
}

// ==================== Memory Allocation Benchmarks ====================

// BenchmarkMemoryAllocation benchmarks memory allocations during GA run
func BenchmarkMemoryAllocation(b *testing.B) {
	population := make([]Chromosome, 100)
	for i := range population {
		population[i] = &MockChromosome{fitness: float64(i)}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		pop := make([]Chromosome, len(population))
		copy(pop, population)
		b.StartTimer()

		ga := New(
			WithPopulation(pop),
			WithGenerations(10),
			WithRandomSeed(12345),
		)
		_ = ga.Run()
	}
}

// BenchmarkTSPMemoryAllocation benchmarks TSP-specific memory allocations
func BenchmarkTSPMemoryAllocation(b *testing.B) {
	cities := make([]City, 50)
	for i := range cities {
		cities[i] = City{
			Name: fmt.Sprintf("City%d", i),
			X:    float64(i * 10),
			Y:    float64(i * 15),
		}
	}

	population := make([]Chromosome, 100)
	for i := range population {
		route := make([]City, len(cities))
		copy(route, cities)
		population[i] = &TSPChromosome{Route: route}
	}

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		b.StopTimer()
		pop := make([]Chromosome, len(population))
		for j := range pop {
			pop[j] = population[j].Clone()
		}
		b.StartTimer()

		ga := New(
			WithPopulation(pop),
			WithGenerations(10),
			WithMutationRate(0.02),
			WithCrossoverRate(0.85),
			WithRandomSeed(12345),
		)
		_ = ga.Run()
	}
}
