package shared;
/**
 * A convergence trainer trains a network
 * until convergence, using another trainer
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ConvergenceTrainer implements Trainer {
    /** The default threshold */
    private static final double THRESHOLD = 0.5;
    /** The maximum number of iterations */
    private static final int MAX_ITERATIONS = 10000;
    /** The mainimum number of iterations */
    private static final int MIN_ITERATIONS = 5000;
    /** The minimum number of unchanged iterations */
    private static final int MIN_UNCHANGED_ITERATIONS = 10;

    /**
     * The trainer
     */
    private Trainer trainer;

    /**
     * The threshold
     */
    private double threshold;
    
    /**
     * The number of iterations trained
     */
    private int iterations;
    
    /**
     * The maximum number of iterations to use
     */
    private int maxIterations;
    
    private int minIterations;

    /**
     * Create a new convergence trainer
     * @param trainer the thrainer to use
     * @param threshold the error threshold
     * @param maxIterations the maximum iterations
     */
    public ConvergenceTrainer(Trainer trainer,
            double threshold, int maxIterations, int minIterations) {
        this.trainer = trainer;
        this.threshold = threshold;
        this.maxIterations = maxIterations;
        this.minIterations = minIterations;
    }
    

    /**
     * Create a new convergence trainer
     * @param trainer the trainer to use
     */
    public ConvergenceTrainer(Trainer trainer) {
        this(trainer, THRESHOLD, MAX_ITERATIONS, MIN_ITERATIONS);
    }

    /**
     * @see Trainer#train()
     */
    public double train() {
    	int countSmallChanges = 0;
        double lastError;
        double error = Double.MAX_VALUE;
        do {
           iterations++;
           lastError = error;
           error = trainer.train();
           
//           System.out.println("iteration " + iterations + ": " + error);
           
           if (Math.abs(lastError - error) < threshold) {
        	   countSmallChanges++;
           } else {
        	   countSmallChanges = 0;
           }
        } while (Math.abs(lastError - error) > threshold
             && iterations < maxIterations
             || iterations < minIterations
             || countSmallChanges < MIN_UNCHANGED_ITERATIONS);
        return error;
    }
    
    /**
     * Get the number of iterations used
     * @return the number of iterations
     */
    public int getIterations() {
        return iterations;
    }
    

}
