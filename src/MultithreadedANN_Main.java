import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.specific.CSVNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.obj.SerializeObject;

public class MultithreadedANN_Main {
	private static int hidden_neurons;
	private static double accuracy;
	private static int max_step;
	private static String file_train;
	private static String file_test;
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		if (args.length != 5) {
			System.out.println("args needed: hidden neurons, accuracy, max_step, file_train, file_test");
		}
		hidden_neurons = Integer.valueOf(args[0]); //48
		accuracy = Double.valueOf(args[1]); //0.01
		max_step = Integer.valueOf(args[2]); //0.01
		file_train = args[3];
		file_test = args[4];

		MultithreadedANN_Main eegNet = new MultithreadedANN_Main();
		//step 1
		eegNet.trainAndSave();
		//step 2
		eegNet.loadAndEvaluate();
		Encog.getInstance().shutdown();
		System.exit(0);
	}

	public void trainAndSave() throws IOException  {
		//make a multilayer perceptron with sigmoid activation functions and bias neurons
		BasicNetwork nnet = new BasicNetwork();
		nnet.addLayer(new BasicLayer(null, true, 84)); //84 inputs (16 sensors * 5 for window + 14 for raw)
		nnet.addLayer(new BasicLayer(new ActivationSigmoid(), true, hidden_neurons));
		nnet.addLayer(new BasicLayer(new ActivationSigmoid(), false, 12)); //12 outputs (facial exprs)
		nnet.getStructure().finalizeStructure();
		nnet.reset();
		//read in the CSV
		// local
		//CSVNeuralDataSet trainingSet = new CSVNeuralDataSet("/Users/x/Desktop/mega.csv", 84, 12, false); //84 in, 12 out
		// UConn HORNET cluster
		CSVNeuralDataSet trainingSet = new CSVNeuralDataSet("/home/amd11037/" + file_train + ".csv", 84, 12, false); //84 in, 12 out
		//train the network
		final ResilientPropagation train = new ResilientPropagation(nnet, trainingSet); //use RPROP for now
		//train.setBatchSize(0);
		System.out.println("Training...");
		long start = System.nanoTime(); //time the training
		long k = 0;
		do {
			train.iteration();
			k++;
			System.out.println(train.getError());
		} while (k < max_step && train.getError() > accuracy);
		train.finishTraining();
		long end = System.nanoTime();
		long elapsedTime = end - start;

		double seconds = (double)elapsedTime / 1000000000.0;
		System.out.println("Training time: " + seconds + " seconds");

		double error = nnet.calculateError(trainingSet);
		System.out.println("Network trained to error: " + error);

		SerializeObject.save(new File("ann_for_"+hidden_neurons+"_neurons_trained_on_"+ file_train +".ser"), nnet);	
		System.out.println("Network saved.");
	}

	public void loadAndEvaluate () throws ClassNotFoundException, IOException
	{
		System.out.println("Loading network...");
		BasicNetwork nnet = (BasicNetwork)SerializeObject.load(new File("ann_for_"+hidden_neurons+"_neurons_trained_on_"+ file_train +".ser"));
		// local
		//CSVNeuralDataSet testSet = new CSVNeuralDataSet("/Users/x/Desktop/SHORTMod.csv", 84, 12, false);
		// UConn HORNET cluster
		CSVNeuralDataSet testSet = new CSVNeuralDataSet("/home/amd11037/" + file_test + ".csv", 84, 12, false);

		double error = nnet.calculateError(testSet);
		System.out.println("Loaded network's error: " + error);
		String file =  file_test + "_ann_res_for_"+hidden_neurons+"_neurons_trained_on_"+ file_train +".csv";
		final PrintWriter writer = new PrintWriter(new FileOutputStream(file, false));
		//System.out.println("Saving data to: " + file + "...");
		//close file on abrupt program termination
		Runtime.getRuntime().addShutdownHook(new Thread() {
			public void run() {
				writer.close();
			}
		});
		for (MLDataPair pair : testSet) {
			//StringJoiner joiner = new StringJoiner(",");
			final MLData output = nnet.compute(pair.getInput());
			//pair.getInput().getData(0) + "," 
			//+ String.valueOf(pair.getInput().getData(1)) + ","
			//+ 
			writer.println((double)Math.round(output.getData(0) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(0)) + ","

					+ (double)Math.round(output.getData(1) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(1)) + ","

					+ (double)Math.round(output.getData(2) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(2)) + ","

					+ (double)Math.round(output.getData(3) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(3)) + ","

					+ (double)Math.round(output.getData(4) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(4)) + ","

					+ (double)Math.round(output.getData(5) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(5)) + ","

					+ (double)Math.round(output.getData(6) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(6)) + ","

					+ (double)Math.round(output.getData(7) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(7)) + ","

					+ (double)Math.round(output.getData(8) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(8)) + ","

					+ (double)Math.round(output.getData(9) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(9)) + ","

					+ (double)Math.round(output.getData(10) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(10)) + ","

					+ (double)Math.round(output.getData(11) * 100000) / 100000 + ","
					+ String.valueOf((int) pair.getIdeal().getData(11)));

		}
		writer.close();
		System.out.println("Done testing for accuracy "+ accuracy);
	}
}
