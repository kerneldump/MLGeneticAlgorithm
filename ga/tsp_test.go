package ga

import (
	"math"
	"os"
	"testing"
)

// TestTSPCrossoverPreservesAllCities verifies Order Crossover (OX1) correctness
func TestTSPCrossoverPreservesAllCities(t *testing.T) {
	cities := []City{
		{Name: "A", X: 0, Y: 0},
		{Name: "B", X: 1, Y: 1},
		{Name: "C", X: 2, Y: 2},
		{Name: "D", X: 3, Y: 3},
		{Name: "E", X: 4, Y: 4},
	}

	parent1 := &TSPChromosome{Route: []City{cities[0], cities[1], cities[2], cities[3], cities[4]}}
	parent2 := &TSPChromosome{Route: []City{cities[4], cities[3], cities[2], cities[1], cities[0]}}

	// Run crossover multiple times
	for i := 0; i < 100; i++ {
		child := parent1.Crossover(parent2).(*TSPChromosome)

		// Verify all cities are present exactly once
		cityCount := make(map[string]int)
		for _, city := range child.Route {
			cityCount[city.Name]++
		}

		if len(cityCount) != len(cities) {
			t.Errorf("Crossover iteration %d: expected %d unique cities, got %d", i, len(cities), len(cityCount))
		}

		for name, count := range cityCount {
			if count != 1 {
				t.Errorf("Crossover iteration %d: city %s appears %d times, expected 1", i, name, count)
			}
		}
	}
}

// TestTSPMutationPreservesAllCities verifies mutation doesn't lose cities
func TestTSPMutationPreservesAllCities(t *testing.T) {
	cities := []City{
		{Name: "A", X: 0, Y: 0},
		{Name: "B", X: 1, Y: 1},
		{Name: "C", X: 2, Y: 2},
		{Name: "D", X: 3, Y: 3},
	}

	chromosome := &TSPChromosome{Route: make([]City, len(cities))}
	copy(chromosome.Route, cities)

	// Mutate multiple times
	for i := 0; i < 50; i++ {
		chromosome.Mutate()

		// Verify all cities still present
		cityCount := make(map[string]int)
		for _, city := range chromosome.Route {
			cityCount[city.Name]++
		}

		if len(cityCount) != len(cities) {
			t.Errorf("Mutation iteration %d: expected %d unique cities, got %d", i, len(cities), len(cityCount))
		}

		for name, count := range cityCount {
			if count != 1 {
				t.Errorf("Mutation iteration %d: city %s appears %d times, expected 1", i, name, count)
			}
		}
	}
}

// TestTSPCloneIndependence verifies cloned chromosomes are independent
func TestTSPCloneIndependence(t *testing.T) {
	original := &TSPChromosome{
		Route: []City{
			{Name: "A", X: 0, Y: 0},
			{Name: "B", X: 1, Y: 1},
		},
	}

	clone := original.Clone().(*TSPChromosome)

	// Modify clone
	clone.Route[0].Name = "Modified"

	// Verify original unchanged
	if original.Route[0].Name != "A" {
		t.Errorf("Clone modification affected original: expected 'A', got '%s'", original.Route[0].Name)
	}
}

// TestTSPFitnessCalculation verifies fitness calculation
func TestTSPFitnessCalculation(t *testing.T) {
	tests := []struct {
		name           string
		route          []City
		expectedResult string // "zero", "positive", or "infinity"
	}{
		{
			name: "normal route",
			route: []City{
				{Name: "A", X: 0, Y: 0},
				{Name: "B", X: 3, Y: 4}, // Distance 5
			},
			expectedResult: "positive",
		},
		{
			name: "zero distance route",
			route: []City{
				{Name: "A", X: 1, Y: 1},
				{Name: "B", X: 1, Y: 1},
			},
			expectedResult: "infinity",
		},
		{
			name:           "single city",
			route:          []City{{Name: "A", X: 0, Y: 0}},
			expectedResult: "zero",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			chromosome := &TSPChromosome{Route: tt.route}
			fitness := chromosome.Fitness()

			switch tt.expectedResult {
			case "zero":
				if fitness != 0 {
					t.Errorf("Expected fitness 0, got %f", fitness)
				}
			case "positive":
				if fitness <= 0 {
					t.Errorf("Expected positive fitness, got %f", fitness)
				}
			case "infinity":
				if fitness != math.Inf(1) {
					t.Errorf("Expected +Inf fitness, got %f", fitness)
				}
			}
		})
	}
}

// TestVisualizeTSP verifies visualization doesn't crash
func TestVisualizeTSP(t *testing.T) {
	route := []City{
		{Name: "A", X: 0, Y: 0},
		{Name: "B", X: 10, Y: 10},
		{Name: "C", X: 20, Y: 5},
	}

	// Test with valid route
	err := VisualizeTSP(route, "test_route.svg")
	if err != nil {
		t.Errorf("VisualizeTSP failed: %v", err)
	}

	// Test with empty route
	err = VisualizeTSP([]City{}, "empty_route.svg")
	if err == nil {
		t.Error("Expected error for empty route, got nil")
	}

	// Cleanup
	_ = os.Remove("test_route.svg")
	_ = os.Remove("empty_route.svg")
}
