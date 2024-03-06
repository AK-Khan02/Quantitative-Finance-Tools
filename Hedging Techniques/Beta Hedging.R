# Placeholder for the beta calculation of the option/portfolio
betaformula <- function(S, K, T, t, r, sig, market_beta){
  # Simplified assumption: beta changes linearly with S
  beta_base <- 0.5 # Assumed base beta of the option/portfolio
  beta_sensitivity <- 0.05 # Assumed sensitivity of beta to stock price changes
  return(beta_base + beta_sensitivity * (S - 100) / 100)
}

# Modified function to include beta hedging
gen_one_valuepath_beta_hedged <- function(r, mu, sig, K, S, T, N, market_beta, market_mu){
  portfolval <- rep(0, N)
  market_position <- rep(0, N) # Position in market index for beta hedging
  if(measure == "Q") mu <- r
  curr_delta <- deltaformula(S = S, K = K, T = T, t = 0, r = r, sig = sig)
  curr_beta <- betaformula(S, K, T, 0, r, sig, market_beta)
  curr_cash  <- bsformula(S = S, K = K, T = T, t = 0, r = r, sig = sig) - curr_delta * S - curr_beta * market_position[1]
  curr_stock <- S
  market_position[1] <- -curr_beta * curr_stock / market_beta # Initial market position to hedge beta

  for(j in 1:(N-1)){
    next_stock <- curr_stock * exp((mu - sig^2/2) * T/N + sig * sqrt(T/N) * rnorm(1))
    next_market_position <- market_position[j] * exp((market_mu - sig^2/2) * T/N + sig * sqrt(T/N) * rnorm(1))
    next_delta <- deltaformula(S = next_stock, K = K, T = T, t = j*T/N, r = r, sig = sig)
    next_beta <- betaformula(next_stock, K, T, j*T/N, r, sig, market_beta)
    next_cash <- curr_cash * exp(r*T/N) + (curr_delta - next_delta) * next_stock + (curr_beta - next_beta) * next_market_position
    portfolval[j] <- next_stock * next_delta + next_cash + next_market_position * next_beta - bsformula(
      S = next_stock, K = K, T = T, t = j*T/N, r = r, sig = sig)
    curr_delta <- next_delta
    curr_beta <- next_beta
    curr_cash  <- next_cash
    curr_stock <- next_stock
    market_position[j+1] <- -curr_beta * curr_stock / market_beta # Adjust market position for beta hedging
  }

  next_stock <- curr_stock * exp((mu - sig^2/2) * T/N + sig * sqrt(T/N) * rnorm(1))
  next_market_position <- market_position[N] * exp((market_mu - sig^2/2) * T/N + sig * sqrt(T/N) * rnorm(1))
  portfolval[N] <- next_stock * curr_delta + curr_cash * exp(r*T/N) + next_market_position * curr_beta - max(0, next_stock - K)
  return(portfolval)
}

# Example usage of the modified function
set.seed(1)
n <- 10000
r <- 0.05
mu <- 0.15
market_mu <- 0.1 # Mean return of the market index
sig <- 0.1
market_beta <- 1 # Beta of the market index
K <- 105
S <- 100
T <- 1
N <- 10 # 10 times trading

out <- sapply(1:n, function(i) gen_one_valuepath_beta_hedged(
  r = r, mu = mu, sig = sig, K = K, S = S , T = T, N = N, market_beta = market_beta, market_mu = market_mu))

# You can then plot and analyze 'out' as in the original script
# Plot a sample of paths from the simulation
plot(NA, xlim = c(0, N), ylim = range(out), xlab = "Time Step", ylab = "Portfolio Value", main = "Sample Paths of Hedged Portfolio")
cols <- rainbow(min(25, ncol(out)))  # Generate colors for the lines
for (i in 1:min(25, ncol(out))) {
  lines(out[, i], col = cols[i])
}
legend("topright", legend = paste("Path", 1:min(25, ncol(out))), col = cols, lty = 1, cex = 0.6)

# Histogram of the final P&L
hist(out[N, ], breaks = 30, main = "Histogram of Final Portfolio Values (P&L)", xlab = "P&L", col = "lightblue")
abline(v = mean(out[N, ]), col = "red", lwd = 2)
text(mean(out[N, ]), max(table(cut(out[N, ], breaks = 30))), labels = "Mean", pos = 3, col = "red")

# Risk measures: Value-at-Risk (VaR) and Expected Shortfall (ES)
sorted_pnl <- sort(out[N, ])
VaR <- sorted_pnl[0.05 * n]  # 95% Value-at-Risk
ES <- mean(sorted_pnl[1:(0.05 * n)])  # Expected Shortfall at 95%
abline(v = VaR, col = "blue", lwd = 2, lty = 2)
text(VaR, max(table(cut(out[N, ], breaks = 30))) * 0.8, labels = "VaR 95%", pos = 3, col = "blue")
abline(v = ES, col = "darkgreen", lwd = 2, lty = 2)
text(ES, max(table(cut(out[N, ], breaks = 30))) * 0.6, labels = "ES 95%", pos = 3, col = "darkgreen")

# Add a legend for the risk measures
legend("topright", inset = 0.05, title = "Risk Measures", cex = 0.8, 
       legend = c("Mean", "VaR 95%", "ES 95%"), 
       col = c("red", "blue", "darkgreen"), 
       lty = c(1, 2, 2), 
       lwd = 2, 
       box.lty = 1)
