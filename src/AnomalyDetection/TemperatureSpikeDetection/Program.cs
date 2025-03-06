using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, World!");


// dotnet add package Microsoft.ML

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Wczytujemy dane
var temperatureData = new List<TemperatureData>
{
       // Normalne dane
    new TemperatureData { Temperature = 20.0f },
    new TemperatureData { Temperature = 21.0f },
    new TemperatureData { Temperature = 19.5f },
    new TemperatureData { Temperature = 22.0f },
    new TemperatureData { Temperature = 20.5f },
    new TemperatureData { Temperature = 21.5f },
    new TemperatureData { Temperature = 19.8f },
    new TemperatureData { Temperature = 20.1f },
    new TemperatureData { Temperature = 21.2f },
    new TemperatureData { Temperature = 19.8f },
    new TemperatureData { Temperature = 19.7f },
    new TemperatureData { Temperature = 22.3f },
    new TemperatureData { Temperature = 20.1f },
    new TemperatureData { Temperature = 21.4f },
    new TemperatureData { Temperature = 19.9f },
    new TemperatureData { Temperature = 20.6f },
    // Anomalie
    new TemperatureData { Temperature = 35.0f }, // Nietypowa wartość
    new TemperatureData { Temperature = -5.0f }, // Nietypowa wartość
    new TemperatureData { Temperature = 40.0f }, // Nietypowa wartość
    new TemperatureData { Temperature = -10.0f } // Nietypowa wartość
};

// 3. Konwersja danych do IDataView
var dataView = context.Data.LoadFromEnumerable(temperatureData);

// 4. Tworzenie pipeline'a z użyciem IidSpikeDetector
// Uwaga: Parametr pvalueHistoryLength powinien odpowiadać liczbie poprzednich obserwacji używanych do obliczeń.

var pipeline = context.Transforms.DetectIidSpike(
    outputColumnName: nameof(TemperaturePrediction.Prediction),
    inputColumnName: nameof(TemperatureData.Temperature),
    confidence: 95d, // Poziom ufności definiuje, jak "pewny" musi być model, aby uznać punkt za anomalię. Zakres 0..100 
    pvalueHistoryLength: temperatureData.Count / 2);

// 5. Trenowanie modelu
var model = pipeline.Fit(dataView);

// 6. Przekształcenie danych i predykcja
var predictions = model.Transform(dataView);

// 7. Pobranie wyników predykcji
var predictionResults = context.Data.CreateEnumerable<TemperaturePrediction>(predictions, reuseRowObject: false);

// 8. Wyświetlenie wyników
Console.WriteLine("Wyniki detekcji anomalii (IidSpikeDetector):");
int index = 0;
foreach (var (input, prediction) in temperatureData.Zip(predictionResults, (d, p) => (d, p)))
{
    // Wektor Prediction zawiera 3 wartości: [Alert, P-Value, Score]
    if (prediction.Alert)
        Console.BackgroundColor = ConsoleColor.Red;

    Console.WriteLine($"Przykład {index + 1}: Temp = {input.Temperature:F1}°C, PValue = {prediction.PValue:F2}, Score = {prediction.Score:F2}, Alert = {prediction.Alert}");
    index++;

    Console.ResetColor();
}

Console.ReadLine();


// Klasa wejściowa 
public class TemperatureData
{
    public float Temperature { get; set; }
}

// Klasa wyjściowa dla predykcji
public class TemperaturePrediction
{
    // Wynik z IidSpikeDetector to wektor 3-wymiarowy: [Alert, P-Value, Score]
    [VectorType(3)]
    public double[] Prediction { get; set; }

    // Wygodne właściwości do odczytu wyników
    public double PValue => Prediction[1];
    public double Score => Prediction[2];
    public bool Alert => Prediction[0] == 1;
}