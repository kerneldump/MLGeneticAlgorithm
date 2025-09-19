package ga

import (
	"fmt"
	"math"
	"os"
)

// VisualizeTSP generates an SVG visualization of a TSP route.
func VisualizeTSP(route []City, filename string) error {
	if len(route) == 0 {
		return fmt.Errorf("empty route")
	}

	// Calculate bounds and scaling
	minX, maxX := route[0].X, route[0].X
	minY, maxY := route[0].Y, route[0].Y

	for _, city := range route {
		if city.X < minX {
			minX = city.X
		}
		if city.X > maxX {
			maxX = city.X
		}
		if city.Y < minY {
			minY = city.Y
		}
		if city.Y > maxY {
			maxY = city.Y
		}
	}

	// Add padding and set canvas size
	padding := 80.0
	canvasWidth := 800.0
	canvasHeight := 600.0

	// Calculate scaling factors
	scaleX := (canvasWidth - 2*padding) / (maxX - minX)
	scaleY := (canvasHeight - 2*padding) / (maxY - minY)
	scale := math.Min(scaleX, scaleY)

	// Function to transform coordinates
	transformX := func(x float64) float64 {
		return padding + (x-minX)*scale
	}
	transformY := func(y float64) float64 {
		return padding + (y-minY)*scale
	}

	// Start building SVG
	svg := fmt.Sprintf(`<svg width="%.0f" height="%.0f" xmlns="http://www.w3.org/2000/svg">`, canvasWidth, canvasHeight)
	svg += `<defs>`
	svg += `<marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">`
	svg += `<polygon points="0 0, 10 3.5, 0 7" fill="blue" />`
	svg += `</marker>`
	svg += `</defs>`

	// Draw lines with arrows between consecutive cities
	for i := 0; i < len(route); i++ {
		current := route[i]
		next := route[(i+1)%len(route)] // Use modulo to connect last city to first

		x1 := transformX(current.X)
		y1 := transformY(current.Y)
		x2 := transformX(next.X)
		y2 := transformY(next.Y)

		// Calculate direction vector and normalize
		dx := x2 - x1
		dy := y2 - y1
		length := math.Sqrt(dx*dx + dy*dy)

		if length > 0 {
			// Adjust line endpoints to not overlap with circles
			circleRadius := 6.0
			offsetX := dx / length * circleRadius
			offsetY := dy / length * circleRadius

			lineX1 := x1 + offsetX
			lineY1 := y1 + offsetY
			lineX2 := x2 - offsetX
			lineY2 := y2 - offsetY

			svg += fmt.Sprintf(`<line x1="%.2f" y1="%.2f" x2="%.2f" y2="%.2f" stroke="blue" stroke-width="2" marker-end="url(#arrowhead)" />`,
				lineX1, lineY1, lineX2, lineY2)
		}
	}

	// Draw cities as circles
	for _, city := range route {
		x := transformX(city.X)
		y := transformY(city.Y)
		svg += fmt.Sprintf(`<circle cx="%.2f" cy="%.2f" r="6" fill="red" stroke="black" stroke-width="1" />`, x, y)
	}

	// Add city labels with names and coordinates
	for _, city := range route {
		x := transformX(city.X)
		y := transformY(city.Y)

		// Position text above the circle
		textY := y - 12

		// City name
		svg += fmt.Sprintf(`<text x="%.2f" y="%.2f" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" font-weight="bold" fill="black">%s</text>`,
			x, textY, city.Name)

		// Coordinates below the name
		coordY := textY - 14
		svg += fmt.Sprintf(`<text x="%.2f" y="%.2f" text-anchor="middle" font-family="Arial, sans-serif" font-size="10" fill="gray">(%.1f,%.1f)</text>`,
			x, coordY, city.X, city.Y)
	}

	// Add title
	titleY := 25.0
	svg += fmt.Sprintf(`<text x="%.2f" y="%.2f" text-anchor="middle" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="black">TSP Route Visualization</text>`,
		canvasWidth/2, titleY)

	// Calculate and display total distance
	totalDistance := 0.0
	for i := 0; i < len(route); i++ {
		current := route[i]
		next := route[(i+1)%len(route)]
		dx := current.X - next.X
		dy := current.Y - next.Y
		totalDistance += math.Sqrt(dx*dx + dy*dy)
	}

	distanceY := canvasHeight - 15
	svg += fmt.Sprintf(`<text x="%.2f" y="%.2f" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="black">Total Distance: %.2f</text>`,
		canvasWidth/2, distanceY, totalDistance)

	svg += `</svg>`

	return os.WriteFile(filename, []byte(svg), 0644)
}
