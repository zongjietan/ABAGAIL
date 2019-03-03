package HW2;

import java.util.Arrays;
import java.util.concurrent.TimeUnit;

import opt.DiscreteChangeOneNeighbor;
import opt.GenericHillClimbingProblem;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.FourPeaksEvaluationFunction;
import opt.example.TwoColorsEvaluationFunction;
import opt.example.CountOnesEvaluationFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.SingleCrossOver;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;
import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;


public class AlgoComparison {
	
	private static int N = 50;
	private static int T = N/5;
	private static int[] ranges = new int[N];
	private static long startTime, endTime, timeElapsed;
	
//	private static FourPeaksEvaluationFunction ef;
	private static CountOnesEvaluationFunction ef;
//	private static TwoColorsEvaluationFunction ef;
	private static DiscreteUniformDistribution odd;
	private static DiscreteChangeOneNeighbor nf;
	private static GenericHillClimbingProblem hcp;
	private static DiscreteChangeOneMutation mf;
	private static SingleCrossOver cf;
	private static GenericGeneticAlgorithmProblem gap;
	private static DiscreteDependencyTree df;
	private static GenericProbabilisticOptimizationProblem pop;
	
	private static void setN(int n) {
		N = n;
	}
	
	private static void run_RHC(int numRestarts) {
		double optimum = 0.0;
		int numIterations = 0;
		
		for (int i=0; i<numRestarts; i++) {
			RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
			/*FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 20000);*/
			ConvergenceTrainer ct = new ConvergenceTrainer(rhc);
			ct.train();
			
			double candidate = ef.value(rhc.getOptimal());
			
			numIterations += ct.getIterations();
			
			if (candidate > optimum) {
				optimum = candidate;
			}
		}
		numIterations /= numRestarts;
		
		System.out.println("RHC: " + optimum);
		System.out.println("Average number of iterations: " + numIterations);
	}
	
	private static void run_SA(double t, double cooling) {
		SimulatedAnnealing sa = new SimulatedAnnealing(t, cooling, hcp);
		/*FixedIterationTrainer fit = new FixedIterationTrainer(sa, 1000000);*/
		ConvergenceTrainer ct = new ConvergenceTrainer(sa);
		ct.train();
		
		double optimum = ef.value(sa.getOptimal());
		int numIterations = ct.getIterations();
		
		System.out.println("SA: " + optimum);
		System.out.println("Number of iterations: " + numIterations);
	}

	
	private static void run_GA(int populationSize, int toMate, int toMutate) {
		StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(populationSize, toMate, toMutate, gap);
//		FixedIterationTrainer fit = new FixedIterationTrainer(ga, 1000);
//		fit.train();
		ConvergenceTrainer ct = new ConvergenceTrainer(ga);
		ct.train();
		
		double optimum = ef.value(ga.getOptimal());
		int numIterations = ct.getIterations();
		
		System.out.println("GA: " + optimum);
		System.out.println("Number of iterations: " + numIterations);
	}
	
	private static void run_MIMIC(int samples, int toKeep) {
		MIMIC mimic = new MIMIC(samples, toKeep, pop);
//		FixedIterationTrainer fit = new FixedIterationTrainer(mimic, 1000);
//		fit.train();
		ConvergenceTrainer ct = new ConvergenceTrainer(mimic);
		ct.train();
		
		double optimum = ef.value(mimic.getOptimal());
		int numIterations = ct.getIterations();
		
		System.out.println("MIMIC: " + optimum);
		System.out.println("Number of iterations: " + numIterations);
	}
	
	
	public static void main(String[] args) throws InterruptedException {
		
		Arrays.fill(ranges, 2);
		
//		ef = new TwoColorsEvaluationFunction();
//		ef = new FourPeaksEvaluationFunction(T);
		ef = new CountOnesEvaluationFunction();
		odd = new DiscreteUniformDistribution(ranges);
		nf = new DiscreteChangeOneNeighbor(ranges);
		hcp = new GenericHillClimbingProblem(ef, odd, nf);
		mf = new DiscreteChangeOneMutation(ranges);
		cf = new SingleCrossOver();
		gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
		df = new DiscreteDependencyTree(0.1, ranges);
		pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
		
		/* randomized hill climbing */
		startTime = System.currentTimeMillis();
		
		run_RHC(200);
		
		endTime = System.currentTimeMillis();
		timeElapsed = endTime - startTime;
		System.out.println("Execution time in milliseconds : " + timeElapsed);
		
		/* simulated annealing */
		startTime = System.currentTimeMillis();
		
		run_SA(1E10, 0.95);
		
		endTime = System.currentTimeMillis();
		timeElapsed = endTime - startTime;
		System.out.println("Execution time in milliseconds : " + timeElapsed);
		
		/* Genetic Algorithm */
		startTime = System.currentTimeMillis();
		
		run_GA(200, 100, 5);
		
		endTime = System.currentTimeMillis();
		timeElapsed = endTime - startTime;
		System.out.println("Execution time in milliseconds : " + timeElapsed);
		
		/* MIMIC Algorithm */
		startTime = System.currentTimeMillis();
		
		run_MIMIC(200, 20);
		
		endTime = System.currentTimeMillis();
		timeElapsed = endTime - startTime;
		System.out.println("Execution time in milliseconds : " + timeElapsed);
		
	}
}
