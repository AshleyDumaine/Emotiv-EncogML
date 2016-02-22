import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

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
	private static String file_train;
	private static String file_test;
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		max_step = Integer.valueOf(args[0]);
		file_train = args[1];
		file_test = args[2];
		SVM_Main eegNet = new SVM_Main();
		accuracy = 0.1;//Double.valueOf(args[0]); //0.01
		eegNet.trainAndSave();
		eegNet.loadAndEvaluate();
		Encog.getInstance().shutdown();
		System.exit(0);
	}
	
	public void trainAndSave() throws IOException {
		//create a SVM for classification, change false to true for regression     
        _svm = new SVM(84, false);
        System.out.println(_svm.getKernelType());
		//_svm = new SVM(84, SVMType.SupportVectorClassification, KernelType.Linear);
		//read in the CSV
        // UConn HORNET cluster
		CSVNeuralDataSet trainingSet = new CSVNeuralDataSet("/home/amd11037/" + file_train + ".csv", 84, 1, false); //70 in, 12 out
		// local
		//CSVNeuralDataSet trainingSet = new CSVNeuralDataSet("/Users/x/Desktop/ALLModCOND.csv", 84, 1, false); //70 in, 12 out
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
	
	public void loadAndEvaluate () throws ClassNotFoundException, IOException {
		PersistSVM persistSVM = new PersistSVM();
		_svm = (SVM) persistSVM.read(new FileInputStream("save_svm_for_" + file_train + ".ser"));
		// test the SVM
		// UConn HORNET cluster
		CSVNeuralDataSet testSet = new CSVNeuralDataSet("/home/amd11037/" + file_test + ".csv", 84, 1, false);
		// local
		//CSVNeuralDataSet testSet = new CSVNeuralDataSet("/Users/x/Desktop/mega.csv", 84, 1, false);
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
