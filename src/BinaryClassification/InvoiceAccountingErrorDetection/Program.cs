using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, World!");

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();


// 2. Wczytanie danych z pliku csv
var trainData = context.Data.LoadFromTextFile<InvoiceData>("invoices.csv", separatorChar: ',', hasHeader: true);

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(trainData, testFraction: 0.2f);
var trainingData = testTrainSplit.TrainSet;
var testData = testTrainSplit.TestSet;

// 4.  Wybór algorytmu - wybieram Catalog klasyfikacji binarnej 
var trainer = context.BinaryClassification.Trainers.SgdCalibrated();

// 5. Budowanie modelu
var pipeline = context.Transforms.Categorical.OneHotEncoding("EncodedDescription", "Description") // Kodowanie
    .Append(context.Transforms.Categorical.OneHotHashEncoding("MPKEncoded", "MPK"))
    .Append(context.Transforms.Concatenate("Features", "Amount", "EncodedDescription", "MPKEncoded"))
    .Append(trainer); // Kodowanie

// 6. Trenowanie modelu na danych treningowych (TrainSet) 80%
var model = pipeline.Fit(trainingData);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testData);

var metrics = context.BinaryClassification.Evaluate(predictions);



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