using Microsoft.ML;

Console.WriteLine("Hello, NormalizeMinMax!");

// Utworzenie kontekstu
var context = new MLContext();

// Przykładowe dane z wartościami liczbowymi
var data = new NumberData[]
{
    new NumberData { Value = 10 },
    new NumberData { Value = 20 },
    new NumberData { Value = 15 },
    new NumberData { Value = 30 },
    new NumberData { Value = 5 }
};

// Załadowane danych do IDataView
var dataView = context.Data.LoadFromEnumerable(data);

// Zdefiniowanie normalizacji w potoku przetwarzania (normalizujemy kolumnę "Value")
var pipeline = context.Transforms.NormalizeMinMax("NormalizedValue", "Value");

// Dopasowanie potoku 
var transformer = pipeline.Fit(dataView);

// Przetwarzanie danych
var transformedData = transformer.Transform(dataView);

// Konwersja z powrotem z celu wyświetlenia danych
var normalizedData = context.Data.CreateEnumerable<NormalizedNumberData>(transformedData, reuseRowObject: false);

// Wyświetlenie danych
foreach (var item in normalizedData)
{
    Console.WriteLine($"Original Value: {item.Value}, Normalized Value: {item.NormalizedValue}");
}

Console.ReadLine();

// Definicja klas
public class NumberData
{
    public float Value { get; set; }
}

public class NormalizedNumberData : NumberData
{
    public float NormalizedValue { get; set; }
}