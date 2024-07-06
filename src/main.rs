extern crate statrs;

use statrs::distribution::{NegativeBinomial, Discrete};

fn main() {
    let k = 3; // Number of successes
    let p = 0.5; // Probability of success in each trial

    // Create a negative binomial distribution instance
    let neg_binom = NegativeBinomial::new(k as f64, p).unwrap();

    // Example usage
    let n = 10; // Total number of trials
    let probability = neg_binom.pmf(n);

    println!("The probability of getting exactly {} successes out of {} trials with success probability {} is {}", k, n, p, probability);
}