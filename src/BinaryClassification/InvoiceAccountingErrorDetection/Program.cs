using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, World!");

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();


// 2. Wczytanie danych z pliku csv
var trainData = context.Data.LoadFromTextFile<InvoiceData>("invoices.csv", separatorChar: ',', hasHeader: true, allowQuoting: true);

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(trainData, testFraction: 0.2f);
var trainingData = testTrainSplit.TrainSet;
var testData = testTrainSplit.TestSet;

// 4.  Wybór algorytmu - wybieram Catalog klasyfikacji binarnej 
var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

// 5. Budowanie modelu
var pipeline = context
    .Transforms.NormalizeMinMax("NormalizedAmount", "Amount")   // Normalizacja kwoty
    .Append(context.Transforms.Text.FeaturizeText("EncodedDescription", "Description") // Kodowanie
    .Append(context.Transforms.Categorical.OneHotHashEncoding("MPKEncoded", "MPK"))
    .Append(context.Transforms.Concatenate("Features", "NormalizedAmount", "EncodedDescription", "MPKEncoded"))
    .Append(trainer)); // Kodowanie

// 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
var model = pipeline.Fit(trainingData);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testData);

var metrics = context.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"Accuracy: {metrics.Accuracy:P2}"); 
Console.WriteLine($"AUC: {metrics.AreaUnderRocCurve:P2}"); 
Console.WriteLine($"F1 Score: {metrics.F1Score}");

var predictionRows = context.Data.CreateEnumerable<InvoicePrediction>(predictions, reuseRowObject: false);
var trueCount = predictionRows.Count(p => p.PredictedIsIncorrect);
var falseCount = predictionRows.Count(p => !p.PredictedIsIncorrect);
Console.WriteLine($"Predictions True: {trueCount} False: {falseCount}");

// 8. Predykcja dla nowej faktury
var newInvoice = new InvoiceData { Amount = 5000, Description = "Konsultacje prawne", MPK = "005" };


// 9. Tworzenie silnika predykcyjnego
var predictionEngine = context.Model.CreatePredictionEngine<InvoiceData, InvoicePrediction>(model);

// 10. Predykcja poprawności księgowania faktury
var prediction = predictionEngine.Predict(newInvoice);

// 11. Wyświetlenie wyniku predykcji
Console.WriteLine($"Prediction: {prediction.PredictedIsIncorrect}");
Console.WriteLine($"Probalitity: {prediction.Probability:P2}");
Console.WriteLine($"Score: {prediction.Score}");

Console.ReadLine();


public class InvoiceData
{
    [LoadColumn(0)]
    public float InvoiceID { get; set; }

    [LoadColumn(1)]
    public float Amount { get; set; }

    [LoadColumn(2)]
    public string Description { get; set; }

    [LoadColumn(3)]
    public string MPK { get; set; }

    [LoadColumn(4), ColumnName("Label")]
    public bool IsIncorrect { get; set; }
}

public class InvoicePrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedIsIncorrect { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
