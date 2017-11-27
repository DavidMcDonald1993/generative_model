library("devtools")
library("NetHypGeom")

main <- function() {
	G <- read_graph("../data/facebook/0.edges", format="edgelist")
	print (fit_power_law(degree(G)))
}

main()