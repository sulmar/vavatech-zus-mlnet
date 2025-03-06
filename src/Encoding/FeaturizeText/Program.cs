
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("Hello, FeaturizeText!");

// Przykładowe dane
var messages = new TextData[]
{
    new TextData { Text = "Kot biega po podwórku" },
    new TextData { Text = "Pies biega po podwórku" },
    new TextData { Text = "Pies szczeka na kota" },
    new TextData { Text = "Ryba pływa w wodzie" },
    new TextData { Text = "Kotek biega po podwórku" },
};

// 1. Utworzenie kontekstu
var context = new MLContext();

// 2. Załadowanie danych
var dataView = context.Data.LoadFromEnumerable(messages);

// 3. Definicja potoku do przekształcania tekstu
var pipeline = context.Transforms.Text.FeaturizeText("Features", "Text");

/* Metoda FeaturizeText opiera się na:
- Tokenizacji: Rozbija tekst na słowa lub n-gramy (np. "kot biega", "biega po").
- TF-IDF: Przydziela wagi słowom/n-gramom na podstawie ich częstotliwości w dokumencie i rzadkości w całym zbiorze danych.  Wspólne frazy (np. "biega po podwórku") mają większą wagę w podobnych zdaniach.
- Wektoryzacja: Zamienia tekst na wektor liczbowy, który jest następnie używany do obliczenia podobieństwa cosinusowego.
*/

// 4. Przekształcenie danych
var transformer = pipeline.Fit(dataView);
var transformedData = transformer.Transform(dataView);

// 5. Pobranie wyników
var preview = transformedData.Preview();

// 6. Wyodrębnienie wektorów cech
var featureColumn = transformedData.GetColumn<float[]>("Features").ToList();
var originalTexts = messages.Select(m => m.Text).ToList();

// 7. Przykład obliczenia podobieństwa (cosinusowego) między wyrażeniami
for (int i = 0; i < featureColumn.Count; i++)
{
    for (int j = i + 1; j < featureColumn.Count; j++)
    {
        float similarity = CosineSimilarity(featureColumn[i], featureColumn[j]);
        Console.WriteLine($"Podobieństwo między '{originalTexts[i]}' a '{originalTexts[j]}': {similarity:F4}");
    }
}
// Funkcja do obliczania podobieństwa cosinusowego
float CosineSimilarity(float[] vectorA, float[] vectorB)
{
    float dotProduct = 0.0f;
    float normA = 0.0f;
    float normB = 0.0f;

    for (int i = 0; i < vectorA.Length; i++)
    {
        dotProduct += vectorA[i] * vectorB[i];
        normA += vectorA[i] * vectorA[i];
        normB += vectorB[i] * vectorB[i];
    }

    return dotProduct / (float)(Math.Sqrt(normA) * Math.Sqrt(normB));
}

Console.ReadLine();


public class TextData
{
    public string Text { get; set; }
}

