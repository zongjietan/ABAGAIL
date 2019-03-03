package HW2;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.AbstractErrorMeasure;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;
import util.linalg.Vector;

public class NNComparison {

	private static int INPUT_LAYER = 4;
	private static int HIDDEN_LAYER = 5;
	private static int OUTPUT_LAYER = 1;
	private static int TRAINING_ITERATIONS = 200;
	private static Instance[] instances;
	
	private static void initialize_instances() throws FileNotFoundException {
		
		ArrayList<Instance> instances_arraylist = new ArrayList<Instance>();
		
		Scanner scanner = new Scanner(new File("data_banknote_authentication.csv"));
		
		while (scanner.hasNextLine()) {
			String line = scanner.nextLine();
			String[] fields = line.split(",");
			
			int numFeature = fields.length - 1;
			double[] features = new double[numFeature];
			for (int i=0; i<numFeature; i++) {
				features[i] = Double.parseDouble(fields[i]);
			}
			Instance instance = new Instance(features);
			
			int label = Integer.parseInt(fields[numFeature]);
			instance.setLabel(new Instance(label));
			
			instances_arraylist.add(instance);
		}
		
		instances = new Instance[instances_arraylist.size()];
		
		for (int i=0; i<instances_arraylist.size(); i++) {
			instances[i] = instances_arraylist.get(i);
		}

	}
	
	private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, Instance[] instances, AbstractErrorMeasure measure) {
		System.out.println("Error results for " + oaName);
		System.out.println("___________________________");
		
		for (int iteration=0; iteration<TRAINING_ITERATIONS; iteration++) {
			oa.train();
			
			double error = 0.0;
			for (int i=0; i<instances.length; i++) {
				network.setInputValues(instances[i].getData());
				network.run();
				
				Instance output = instances[i].getLabel();
				Vector output_values = network.getOutputValues();
				Instance example = new Instance(output_values, new Instance(output_values.get(0)));
				error += measure.value(output, example);
			}
			
			System.out.format("%.3f ", error);
		}
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		initialize_instances();
		
		BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
		SumOfSquaresError measure = new SumOfSquaresError();
		DataSet data_set = new DataSet(instances);
		
		ArrayList<BackPropagationNetwork> networks = new ArrayList<BackPropagationNetwork>();
		ArrayList<NeuralNetworkOptimizationProblem> nnop = new ArrayList<NeuralNetworkOptimizationProblem>();
		ArrayList<OptimizationAlgorithm> oa = new ArrayList<OptimizationAlgorithm>();
		String[] oa_names = {"RHC", "SA", "GA"};
		int[] nnodes = {INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER};
		
		for (String name: oa_names) {
			BackPropagationNetwork classification_network = factory.createClassificationNetwork(nnodes);
			networks.add(classification_network);
			nnop.add(new NeuralNetworkOptimizationProblem(data_set, classification_network, measure));
		}
		
		RandomizedHillClimbing rhc = new RandomizedHillClimbing(nnop.get(0));
		SimulatedAnnealing sa = new SimulatedAnnealing(1E10, 0.99, nnop.get(1));
		StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, nnop.get(2));
		oa.add(rhc);
		oa.add(sa);
		oa.add(ga);
		
		long startTime, endTime, testing_time, training_time;
		
		for (int i=0; i<oa_names.length; i++) {
			String name = oa_names[i];
			startTime = System.currentTimeMillis();
			int correct = 0, incorrect = 0;
			
			train(oa.get(i), networks.get(i), oa_names[i], instances, measure);
			
			endTime = System.currentTimeMillis();
			training_time = endTime - startTime;
			
			Instance optimal_instance = oa.get(i).getOptimal();
			networks.get(i).setWeights(optimal_instance.getData());
			
			startTime = System.currentTimeMillis();
			for (Instance instance: instances) {
				networks.get(i).setInputValues(instance.getData());
				networks.get(i).run();
				
				double predicted = instance.getLabel().getContinuous();
				double actual = networks.get(i).getOutputValues().get(0);
				
				if (Math.abs(predicted - actual) < 0.5) {
					correct += 1;
				} else {
					incorrect += 1;
				}
			}
			endTime = System.currentTimeMillis();
			testing_time = endTime - startTime;
			
			double percentage = correct/(correct + incorrect) * 100;
			
			System.out.format("%nResults for %s: %nCorrectly classified %d instances.",name, correct);
			System.out.format("%nIncorrectly classified %d instances.%nPercent correctly classified: %.3f", incorrect, percentage);
			System.out.format("%nTraining time: %d milliseconds", training_time);
			System.out.format("%nTesting time: %d milliseconds", testing_time);
		}
		
	}

}
