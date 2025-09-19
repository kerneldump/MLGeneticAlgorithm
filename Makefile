GO ?= go
BIN ?= bin/ga

.PHONY: build install test fmt clean example-run example-run-tsp setup

build:
	@mkdir -p bin
	$(GO) build -o $(BIN) ./cmd/ga

install:
	$(GO) install ./cmd/ga

test:
	$(GO) test ./...

fmt:
	$(GO) fmt ./...

setup:
	@mkdir -p examples

example-run: build
	$(BIN)

example-run-tsp: build setup
	$(BIN) --example=tsp

clean:
	rm -rf bin tsp_route.svg