using Microsoft.ML;
using Microsoft.ML.Data;
using System.Globalization;

Console.WriteLine("Hello, Currency Rate Anomaly Detection!");

// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane z pliku CSV
var rates = context.Data.LoadFromTextFile<CurrencyRateRawData>("archiwum_tab_a_2021.csv", hasHeader: true, separatorChar: ';');

var filteredRates = context.Data.SkipRows(rates, 1);

var preview = filteredRates.Preview();

// 3. Dzielimy dane wejściowe na zbiór treningowy i testowy (20% testowy)
var testTrainSplit = context.Data.TrainTestSplit(filteredRates, testFraction: 0.2f);

// 4. Wybór algorytmu

// dotnet add package Microsoft.ML.TimeSeries
var trainer = context.Transforms.DetectIidSpike(
    outputColumnName: "Prediction", inputColumnName: "EUR", confidence: 95d, pvalueHistoryLength: 50);

// 5. Budowanie modelu

Action<CurrencyRateRawData, CurrencyRateData> mapper = (input, output) =>
{
    output.Date = DateTime.ParseExact(input.Date, "yyyyMMdd", CultureInfo.InvariantCulture);
    output.EUR = input.EUR;
    
};

var pipeline = context.Transforms.CustomMapping(mapper, "DateMapper")
    .Append(trainer);

// 6. Trenowanie modelu 
var model = pipeline.Fit(testTrainSplit.TrainSet);

// 7. Ewaluacja modelu na zbiorze testowym (TestSet) 20%
var predictions = model.Transform(testTrainSplit.TestSet);

var predictionResults = context.Data.CreateEnumerable<CurrencyRatePrediction>(predictions, reuseRowObject: false)
    .ToList();


foreach(var result in predictionResults)
{
    var prediction = result.Prediction;

    var alert = prediction[0];  // 1 = anomalia, 0 = brak anomalii
    var score = prediction[1];  // Wartość kursu
    var pValue = prediction[2]; // p-Value

    if (alert==1)
    {
        Console.BackgroundColor = ConsoleColor.Red;
        Console.ForegroundColor = ConsoleColor.White;
    }

    Console.WriteLine($"{alert} {score} EUR {pValue}");

    Console.ResetColor();

}



Console.ReadLine();

public class CurrencyRateRawData
{
    [LoadColumn(0)]
    public string Date { get; set; }

    [LoadColumn(8)]
    public float EUR { get; set; }
}

public class CurrencyRateData
{
    public DateTime Date { get; set; }
    public float EUR { get; set; }
}


public class CurrencyRatePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; } // [Alert, Score, P-Value]
}