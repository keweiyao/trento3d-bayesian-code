"""
1.	my_mtd module is my testing module for Model-to-Data comparsion in the spirit of Jonah's Baysian analysis paper
2.	package required:
	2.1	pyDO: experimental design package, for Latin-hypercube sampling
	2.2	george: gaussian process package
	2.3 emcee: Markov-Chain Monte-Carlo package
	2.4 scipy: for optimization
	2.5 corner:	for plotting corner plot for high dimension parameter space samplings
"""
if __name__ == "__main__":
	print __doc__

__all__ = ["params", "gp_manager", "data_SVD"]
