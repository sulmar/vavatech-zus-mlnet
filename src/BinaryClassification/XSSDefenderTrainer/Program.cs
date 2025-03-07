// See https://aka.ms/new-console-template for more information
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Data;

Console.WriteLine("Hello, World!");


// 1. Utworzenie kontekstu (Create Context)
MLContext context = new MLContext(seed: 1);

// 2. Ładowanie danych (Load data)
IDataView dataView =
    context.Data.LoadFromTextFile<XssInput>("XSS_dataset.csv", hasHeader: true, separatorChar: ',', allowQuoting: true);

// Podział danych na zbiór treningowy i testowy
var trainTestSplit = context.Data.TrainTestSplit(dataView, testFraction: 0.2f);
var trainingData = trainTestSplit.TrainSet;
var testData = trainTestSplit.TestSet;

var preview = dataView.Preview();

// Przygotowanie trenera
var trainer = context.BinaryClassification.Trainers.SdcaLogisticRegression();

// Utworzenie potoku przetwarzania
var pipeline = context.Transforms
    .Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(XssInput.Sentence))
    .Append(trainer);

// Trenowanie
var model = pipeline.Fit(trainingData);


// Ewaluacja
var predictions = model.Transform(testData);

var metrics = context.BinaryClassification.Evaluate(predictions);

Console.WriteLine($"{metrics.Accuracy:P2}");

context.Model.Save(model, trainingData.Schema, "xss-model.zip");





Console.ReadLine();

public class XssInput
{
    [LoadColumn(1)]
    public string Sentence { get; set; }

    [LoadColumn(2)]
    public bool Label { get; set; }
}