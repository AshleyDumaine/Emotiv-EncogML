import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.StringJoiner;

import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.specific.CSVNeuralDataSet;
import org.encog.ml.svm.KernelType;
import org.encog.ml.svm.PersistSVM;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.SVMType;
import org.encog.ml.svm.training.SVMSearchTrain;
import org.encog.ml.svm.training.SVMTrain;
import org.encog.ml.train.MLTrain;

public class SVM_Main {
	private SVM _svm;
	private static double accuracy;
	private static int max_step;
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		max_step = Integer.valueOf(args[0]);
		String file_train = args[1];
		String file_test = args[2];
		SVM_Main eegNet = new SVM_Main();
		accuracy = 0.1;//Double.valueOf(args[0]); //0.01
		eegNet.flattenData(file_train, "/home/amd11037/" + file_train + "_flattened.csv", "MAXIMUM");
		eegNet.trainAndSave("/home/amd11037/" + file_train + "_flattened.csv");
		eegNet.flattenData(file_test, "/home/amd11037/" + file_test + "_flattened.csv", "MAXIMUM");
		eegNet.loadAndEvaluate("/home/amd11037/" + file_train + "_flattened.csv", "/home/amd11037/" + file_test + "_flattened.csv");
		Encog.getInstance().shutdown();
		System.exit(0);
	}

	// perform locally
	public void flattenData(String inFile, String outFile, String method) {
		// take the mean or max of the bucket of 128 rows for each sensor
		// make a new file so the original isn't screwed up
		BufferedReader br = null;
		String line = "";
		// 14 sensors, 128 rows
		int linecount = 0;
		double[][] table = new double[14][128];
		try {
			br = new BufferedReader(new FileReader(inFile));
			final PrintWriter writer = new PrintWriter(new FileOutputStream(outFile, false));
			while ((line = br.readLine()) != null) {
				linecount++;
				int count = linecount % 128;
				String[] sensorReadings = line.split(",");
				for (int i = 0; i < table.length; i++) {
					table[i][count] = Double.parseDouble(sensorReadings[i]);
				}
				// table is full and ready to have features extracted
				if (count == 127) {
					StringJoiner joiner = new StringJoiner(",");
					double[] sensorRes = new double[14];
					for (int j = 0; j < table.length; j++) {
						if (method.equals("MAXIMUM")) {
							// find max of columns of table
							for (int row = 0; row <= count; row++) {
								sensorRes[j] = (table[j][row] > sensorRes[j]) ? table[j][row] : sensorRes[j];
							}
							// write max to the new csv
							joiner.add(String.valueOf(sensorRes[j]));
						}
						else if (method.equals("MEAN")) {
							// total columns of table
							for (int row = 0; row <= count; row++) {
								sensorRes[j] += table[j][row];
							}
							// write average to the new csv
							joiner.add(String.valueOf(sensorRes[j] / 128));
						}
					}
					String joined = joiner.toString();
					writer.println(joined);
				}
			}
			writer.close();
		} catch (Exception e) {
			e.printStackTrace();
		} 
	}

	public void trainAndSave(String file_train) throws IOException {
		//create a SVM for classification, change false to true for regression     
		_svm = new SVM(14, false);
		//System.out.println(_svm.getKernelType());
		//_svm = new SVM(84, SVMType.SupportVectorClassification, KernelType.Linear);
		// read in the CSV
		// UConn HORNET cluster
		CSVNeuralDataSet trainingSet = new CSVNeuralDataSet("/home/amd11037/" + file_train + ".csv", 14, 1, false); //14 in, 1 (12) out
		// local
		//CSVNeuralDataSet trainingSet = new CSVNeuralDataSet("/Users/x/Desktop/ALLModCOND.csv", 14, 1, false); //14 in, 1 (12) out
		// train the SVM
		final MLTrain train = new SVMSearchTrain(_svm, trainingSet);
		System.out.println("Training...");
		long start = System.nanoTime(); //time the training
		long k = 0;
		do {
			train.iteration();
			k++;
			System.out.println(train.getError());
			PersistSVM persistSVM = new PersistSVM(); //save here and overwrite on next iteration
			persistSVM.save(new FileOutputStream("save_svm_for_" + file_train + ".ser", false), _svm);
		} while (k < max_step && train.getError() > accuracy);
		train.finishTraining();
		long end = System.nanoTime();
		long elapsedTime = end - start;

		double seconds = (double)elapsedTime / 1000000000.0;
		System.out.println("Training time: " + seconds + " seconds");

		double error = _svm.calculateError(trainingSet);
		System.out.println("Network trained to error: " + error);

		PersistSVM persistSVM = new PersistSVM();
		persistSVM.save(new FileOutputStream("save_svm_for_" + file_train + ".ser", false), _svm);
	}

	public void loadAndEvaluate (String file_train, String file_test) throws ClassNotFoundException, IOException {
		PersistSVM persistSVM = new PersistSVM();
		_svm = (SVM) persistSVM.read(new FileInputStream("save_svm_for_" + file_train + ".ser"));
		// test the SVM
		// UConn HORNET cluster
		CSVNeuralDataSet testSet = new CSVNeuralDataSet("/home/amd11037/" + file_test + ".csv", 14, 1, false);
		// local
		//CSVNeuralDataSet testSet = new CSVNeuralDataSet("/Users/x/Desktop/mega.csv", 14, 1, false);
		String file = "svm_res_trained_with_" + file_train + "_tested_with_" + file_test + ".csv";
		final PrintWriter writer = new PrintWriter(new FileOutputStream(file, false));
		System.out.println("Saving data to: " + file + "...");
		//close file on abrupt program termination
		Runtime.getRuntime().addShutdownHook(new Thread() {
			public void run() {
				writer.close();
			}
		});
		for (MLDataPair pair : testSet) {
			final MLData output = _svm.compute(pair.getInput());
			writer.println(/*pair.getInput().getData(0) + "," 
			+ String.valueOf(pair.getInput().getData(1)) + ","
			+ */(double)Math.round(output.getData(0) * 100000) / 100000 + ","
			+ String.valueOf((int) pair.getIdeal().getData(0)));
		}
		System.out.println("Done");
		writer.close();
	}
}
