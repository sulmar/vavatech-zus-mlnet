
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, World!");


// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane

var temperatureData = new List<TemperatureData>
{
    new TemperatureData { Date = new DateTime(2025, 3, 1), Temperature = 15f },
    new TemperatureData { Date = new DateTime(2025, 3, 2), Temperature = 16f },
    new TemperatureData { Date = new DateTime(2025, 3, 3), Temperature = 15f },
    new TemperatureData { Date = new DateTime(2025, 3, 4), Temperature = 16f },
    new TemperatureData { Date = new DateTime(2025, 3, 5), Temperature = 15f },
    new TemperatureData { Date = new DateTime(2025, 3, 6), Temperature = 25f }, // Nagły skok temperatury
    new TemperatureData { Date = new DateTime(2025, 3, 7), Temperature = 16f },
    new TemperatureData { Date = new DateTime(2025, 3, 8), Temperature = 15f },
    new TemperatureData { Date = new DateTime(2025, 3, 9), Temperature = 16f },
    new TemperatureData { Date = new DateTime(2025, 3, 10), Temperature = 15f },
    new TemperatureData { Date = new DateTime(2025, 3, 11), Temperature = 15f }
};

var dataView = context.Data.LoadFromEnumerable(temperatureData);


var estimator = context.Transforms.DetectIidSpike(
    outputColumnName: nameof(TemperaturePrediction.Prediction),
    inputColumnName: nameof(TemperatureData.Temperature),
    confidence: 0.95,
    pvalueHistoryLength: temperatureData.Count / 4);


var emptyData = context.Data.LoadFromEnumerable(new List<TemperatureData>());
var transformer = estimator.Fit(emptyData);

var transformedData = transformer.Transform(dataView);

var predictions = context.Data.CreateEnumerable<TemperaturePrediction>(transformedData, reuseRowObject: false);

int index = 0;

foreach (var prediction in predictions)
{
    

    Console.Write($"{temperatureData[index].Date} {temperatureData[index].Temperature} ");
    

    var alert = prediction.Prediction[0];  // 1 = anomalia, 0 = brak anomalii
    var score = prediction.Prediction[1];  // Wartość temp.
    var pValue = prediction.Prediction[2]; // p-Value

    if (alert == 1)
    {
        Console.BackgroundColor = ConsoleColor.Red;
        Console.ForegroundColor = ConsoleColor.White;
    }

    Console.WriteLine($"{alert} {score:F2} {pValue:F4}");

    Console.ResetColor();


    index++;
}


// Klasa wejściowa 
public class TemperatureData
{
    public DateTime Date { get; set; }
    public float Temperature { get; set; }
}

// Klasa wyjściowa dla predykcji
public class TemperaturePrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; }
}