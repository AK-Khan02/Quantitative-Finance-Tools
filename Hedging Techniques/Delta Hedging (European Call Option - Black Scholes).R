library(parallel) # For parallel computing

# Black-Scholes formula for call option price
bsformula <- function(S, K, T, r, sig) {
  d1 <- (log(S/K) + (r + sig^2 / 2) * T) / (sig * sqrt(T))
  d2 <- d1 - sig * sqrt(T)
  pnorm(d1) * S - pnorm(d2) * K * exp(-r * T)
}

# Delta of a call option
deltaformula <- function(S, K, T, r, sig) {
  d1 <- (log(S/K) + (r + sig^2 / 2) * T) / (sig * sqrt(T))
  pnorm(d1)
}

# Generate one path of portfolio values
gen_one_valuepath <- function(r, mu, sig, K, S0, T, N) {
  dt <- T / N
  S <- c(S0, rep(0, N))
  delta <- c(deltaformula(S0, K, T, r, sig), rep(0, N))
  cash <- c(bsformula(S0, K, T, r, sig) - delta[1] * S0, rep(0, N))
  
  for (j in 2:(N + 1)) {
    S[j] <- S[j - 1] * exp((mu - sig^2 / 2) * dt + sig * sqrt(dt) * rnorm(1))
    delta[j] <- deltaformula(S[j], K, T - (j - 1) * dt, r, sig)
    cash[j] <- cash[j - 1] * exp(r * dt) + (delta[j - 1] - delta[j]) * S[j]
  }
  
  # Portfolio value excluding the final option payoff
  portfolval <- S * delta + cash
  
  # Adjust the final value for the option payoff
  portfolval[N + 1] <- portfolval[N + 1] - max(S[N + 1] - K, 0)
  
  return(portfolval)
}

# Parameters
set.seed(1)
n <- 10000
r <- 0.05
mu <- 0.15
sig <- 0.1
K <- 105
S0 <- 100
T <- 1
N <- 10 # Trading times

# Run simulations in parallel
cl <- makeCluster(detectCores() - 1) # Use one less than all cores
clusterExport(cl, c("gen_one_valuepath", "r", "mu", "sig", "K", "S0", "T", "N", "deltaformula", "bsformula"))
clusterEvalQ(cl, library(stats)) # Load stats package on all worker nodes

out <- parSapply(cl, 1:n, function(i) gen_one_valuepath(r, mu, sig, K, S0, T, N))
stopCluster(cl)

# Plot 25 paths
plot(NA, xlim = c(0, N), ylim = range(out), xlab = "Time Steps", ylab = "Portfolio Value")
for (i in 1:min(25, ncol(out))) {
  lines(out[, i], col = i)
}

# Histogram of final P&L
hist(out[N + 1, ], breaks = 30, main = "Histogram of P&L", xlab = "P&L")
abline(v = mean(out[N + 1, ]), col = "red")

# Risk measures
sortedval <- sort(out[N + 1, ])
VaR <- sortedval[0.05 * n] # Value-at-Risk at 95%
ES <- mean(sortedval[1:(0.05 * n)]) # Expected Shortfall at 95%
abline(v = c(VaR, ES), col = "blue")
