/*

// Metropolis Monte Carlo simulation with Lennard–Jones potential
// CHEM 272 — UC Berkeley MSSE
// Authors: Group 2 (Yash Maheshwaran et al.)
// Implements random-scan and all-at-once Metropolis updates for 2D particles.

*/



#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

// RNG helper struct for random number generation
struct RNG {
    std::mt19937 gen; // Mersenne Twister engine
    std::uniform_real_distribution<double> uni01{0.0, 1.0}; // uniform [0,1)

    explicit RNG(unsigned seed) : gen(seed) {}

    // Uniform random number between a and b
    double uniform(double a, double b) {
        std::uniform_real_distribution<double> d(a, b);
        return d(gen);
    }

    // Randomly return +1.0 or -1.0 with equal probability
    double choice_pm() { return uni01(gen) < 0.5 ? -1.0 : 1.0; }
};

// Generate initial particle positions in a square box of given radius
std::pair<std::vector<double>, std::vector<double>>
initial_positions(double radius, int n, RNG& rng) {
    std::vector<double> x(n), y(n);
    for (int i = 0; i < n; i++) {
        x[i] = rng.uniform(-radius, radius);
        y[i] = rng.uniform(-radius, radius);
    }
    return {x, y};
}

// Generate initial particle masses uniformly between low and high
std::vector<double> initial_masses(int n, double low, double high, RNG& rng) {
    std::vector<double> m(n);
    for (int i = 0; i < n; i++) m[i] = rng.uniform(low, high);
    return m;
}

// Compute Lennard-Jones potential matrix φ[i][j] for all pairs
std::vector<std::vector<double>> lennard_jones_matrix(
    const std::vector<std::vector<double>>& dx,
    const std::vector<std::vector<double>>& dy,
    double a, double b) {
    int n = static_cast<int>(dx.size());
    std::vector<std::vector<double>> phi(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue; // skip self-interaction
            double r2 = dx[i][j] * dx[i][j] + dy[i][j] * dy[i][j];
            phi[i][j] = (a / std::pow(r2, 6)) - (b / std::pow(r2, 3));
        }
    }
    return phi;
}

// Compute total potential energy per particle U[i]
std::vector<double> total_potential_per_particle(
    const std::vector<double>& x,
    const std::vector<double>& y,
    double a, double b) {
    int n = static_cast<int>(x.size());

    // Pairwise coordinate differences
    std::vector<std::vector<double>> dx(n, std::vector<double>(n));
    std::vector<std::vector<double>> dy(n, std::vector<double>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dx[i][j] = x[i] - x[j];
            dy[i][j] = y[i] - y[j];
        }
    }

    // Compute potential matrix
    auto phi = lennard_jones_matrix(dx, dy, a, b);

    // Sum φ[i][j] over j for each i
    std::vector<double> U(n, 0.0);
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += phi[i][j];
        U[i] = sum;
    }
    return U;
}

// Perform one Metropolis sweep using random-scan update
void metropolis_random_scan(
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& U,
    double a, double b, double T, double delta, double box_half, RNG& rng) {
    int n = static_cast<int>(x.size());

    // Create shuffled index order
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    std::shuffle(idx.begin(), idx.end(), rng.gen);

    // Try moving each particle in random order
    for (int i : idx) {
        double dx = delta * rng.choice_pm();
        double dy = delta * rng.choice_pm();

        // Trial positions
        std::vector<double> xt = x;
        std::vector<double> yt = y;
        xt[i] = std::max(-box_half, std::min(box_half, xt[i] + dx));
        yt[i] = std::max(-box_half, std::min(box_half, yt[i] + dy));

        // Compute change in potential for particle i
        double U_new_i = total_potential_per_particle(xt, yt, a, b)[i];
        double dU = U_new_i - U[i];

        // Accept move if energy decreases or with probability exp(-dU/T)
        if (dU < 0.0 || rng.uni01(rng.gen) < std::exp(-dU / T)) {
            x[i] = xt[i];
            y[i] = yt[i];
            U[i] = U_new_i;
        }
    }
}

// Perform one Metropolis sweep updating all particles at once
void metropolis_all_at_once(
    std::vector<double>& x,
    std::vector<double>& y,
    std::vector<double>& U,
    double a, double b, double T, double delta, double box_half, RNG& rng) {
    int n = static_cast<int>(x.size());

    // Propose trial positions for all particles
    std::vector<double> xt(n), yt(n);
    for (int i = 0; i < n; i++) {
        xt[i] = std::max(-box_half, std::min(box_half, x[i] + delta * rng.choice_pm()));
        yt[i] = std::max(-box_half, std::min(box_half, y[i] + delta * rng.choice_pm()));
    }

    // Compute new potential energies for all particles
    auto U_new = total_potential_per_particle(xt, yt, a, b);

    // Accept or reject each particle independently
    for (int i = 0; i < n; i++) {
        double dU = U_new[i] - U[i];
        if (dU < 0.0 || rng.uni01(rng.gen) < std::exp(-dU / T)) {
            x[i] = xt[i];
            y[i] = yt[i];
            U[i] = U_new[i];
        }
    }
}

// Save particle data to CSV file
void save_csv(const std::string& filename,
              const std::vector<double>& x,
              const std::vector<double>& y,
              const std::vector<double>& mass,
              int step, double T, double a, double b) {
    std::ofstream file(filename);
    file << "step,x,y,mass,T,a,b\n";
    for (size_t i = 0; i < x.size(); i++) {
        file << step << "," << x[i] << "," << y[i] << "," << mass[i]
             << "," << T << "," << a << "," << b << "\n";
    }
}

// Simulator class encapsulates state and run loop
struct Simulator {
    int n;                 // number of particles
    double radius;         // initial position radius
    RNG rng;               // random number generator
    std::vector<double> x; // x coordinates
    std::vector<double> y; // y coordinates
    std::vector<double> mass;

    // Constructor: initialize positions and masses
    Simulator(int n_, double radius_, unsigned seed)
        : n(n_), radius(radius_), rng(seed) {
        auto pos = initial_positions(radius, n, rng);
        x = pos.first;
        y = pos.second;
        mass = initial_masses(n, 0.5, 2.0, rng);
    }

    // Run simulation for n_iter steps
    void run(int n_iter, double a, double b, double T, double delta, double box_half,
             int plot_every, const std::string& mode) {
        auto U = total_potential_per_particle(x, y, a, b);

        for (int step = 0; step <= n_iter; step++) {
            // Update positions based on chosen mode
            if (mode == "all_at_once")
                metropolis_all_at_once(x, y, U, a, b, T, delta, box_half, rng);
            else
                metropolis_random_scan(x, y, U, a, b, T, delta, box_half, rng);

            // Save snapshot every plot_every steps
            if (step % plot_every == 0) {
                std::string fname = "../results/particles_step_" + std::to_string(step) + ".csv";
                save_csv(fname, x, y, mass, step, T, a, b);
            }
        }
    }
};

int main() {
    // Create simulator with 200 particles in a box of radius 100, seed RNG with 42
    Simulator sim(200, 100.0, 42);

    // Run 4000 iterations, LJ parameters a=1, b=1, temperature T=25,
    // step size delta=1, box half-width=100, save every 500 steps,
    // use "random_scan" update mode
    sim.run(4000, 1.0, 1.0, 25.0, 1.0, 100.0, 500, "random_scan");

    return 0;
}