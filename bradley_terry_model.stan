//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// bradley_terry_model.stan
data {
  int<lower=1> N;           // no. of games
  int<lower=1> P;           // no. of players
  int<lower=1, upper=P> player1[N];  // player 1 index
  int<lower=1, upper=P> player2[N];  // player 2 index
  int<lower=0, upper=1> outcome[N];  // 1 if player 1 wins, 0 if player 2 wins
}

parameters {
  real mu;                  // global mean of log abilities
  real<lower=0> tau;        // global shrinkage parameter
  vector<lower=0>[P] lambda; // local shrinkage parameters
  vector[P] log_theta_raw;  // unshrunk log abilities
}

transformed parameters {
  vector[P] log_theta;      // shrunk log abilities
  log_theta = mu + lambda .* log_theta_raw * tau; 

  vector[P] theta;          //  abilities on the original scale
  theta = exp(log_theta);   // exponentiating to get back to original scale
}

model {
  mu ~ normal(0, 1);        // prior on global mean of log abilities
  tau ~ normal(0,0.5) T[0,];       // global shrinkage parameter
  lambda ~ normal(0, 0.5) T[0,];    // local shrinkage parameters
  log_theta_raw ~ normal(0, 1); // prior on unshrunk log abilities

  for (n in 1:N) {
    outcome[n] ~ bernoulli_logit(log_theta[player1[n]] - log_theta[player2[n]]);
  }
}
