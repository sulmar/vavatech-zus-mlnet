using Microsoft.ML;
using Microsoft.ML.Data;
using Tesseract;

Console.WriteLine("Hello, OCR Tesseract + Classification ML.NET!");

// Instalacja Tesseract 
// 1.Pobierz binaria Tesseract OCR dla Windows:
// https://github.com/UB-Mannheim/tesseract/wiki
// 2. Pobierz pliki językowe
// Pobierz binaria Tesseract OCR dla Windows:
// https://github.com/UB-Mannheim/tesseract/wiki
// 3. Zainstaluj je i zapamiętaj ścieżkę (np. C:\Program Files\Tesseract-OCR).
// 4.Pobierz pliki językowe (.traineddata) dla języka polskiego lub innego, np.:
// Polski: https://github.com/tesseract-ocr/tessdata/blob/main/pol.traineddata
// Angielski: https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata
// Umieść pliki w katalogu tessdata Tesseract OCR.

//  Zainstaluj pakiet NuGet
// dotnet add package Tesseract

string tessDataPath = @"C:\Program Files\Tesseract-OCR\tessdata"; // Ścieżka do plików językowych
string trainingImagesPath = @"C:\training-images";  // Ścieżka do dokumentów do nauki
string newImagePath = "new-document.jpg";           // Ścieżka do dokumentu do predykcji

using var engine = new TesseractEngine(tessDataPath, "pol", EngineMode.Default);

// Pobieramy listę plików
var files = Directory.GetFiles(trainingImagesPath, "*.jpg", SearchOption.AllDirectories);

List<Document> documents = new();

// Dokonujemy ekstracji tekstu
foreach (var imagePath in files)
{
    using var image = Pix.LoadFromFile(imagePath);
    using var page = engine.Process(image);
        
    string extractedText = page.GetText();
    
    Console.WriteLine("Rozpoznany tekst:");
    Console.WriteLine(extractedText);
    
    // Tworzymy dokument
    var document = new Document() { Content = extractedText, Area = "A" };  
    documents.Add(document);
}

// 1. Tworzymy kontekst ML.NET
var context = new MLContext();

// 2. Ladujemy dane
var trainingData = context.Data.LoadFromEnumerable(documents);

// 3. Wybór algorytm
var trainer = context.MulticlassClassification.Trainers.SdcaMaximumEntropy();

// 4. Budujemy model
var mapValueToKey = context.Transforms.Conversion.MapValueToKey(inputColumnName: "Label", outputColumnName: "Label");
var featuredTitle = context.Transforms.Text.FeaturizeText(outputColumnName: "FeaturedContent", inputColumnName: "Content");
var features = context.Transforms.Concatenate( "Features", "FeaturedContent");
var mapKeyValue = context.Transforms.Conversion.MapKeyToValue(outputColumnName: "PredictedLabel", inputColumnName: "PredictedLabel");

var pipeline = mapValueToKey
    .Append(featuredTitle)
    .Append(features)
    .Append(trainer)
    .Append(mapKeyValue);

// 5. Trenowanie modelu na zOCRowanych dokumentach
var model = pipeline.Fit(trainingData);

// 6. Utworzenie silnika predykcyjnego
var predictionEngine = context.Model.CreatePredictionEngine<Document, DocumentPrediction>(model);


// 7. OCR nowego dokumentu
using var newImage = Pix.LoadFromFile(newImagePath);
using var newPage = engine.Process(newImage);
        
string newExtractedText = newPage.GetText();
Console.WriteLine("Rozpoznany tekst:");
Console.WriteLine(newExtractedText);

// 7. Predykcja nowego dokumentu - przydzielenie do obszaru na podstawie zawartości tekstu
var newDocument = new Document() { Content = newExtractedText  }; 

var result = predictionEngine.Predict(newDocument);

Console.WriteLine($"Przydzielono do obszaru: {result.PredictedArea}");

Console.ReadKey();

// Klasa reprezentująca dokument
public class Document
{
    public string Content { get; set; }

    [ColumnName("Label")]
    public string Area { get; set; }
}



// Klasa wynikowa predykcji klastrowania
public class DocumentPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedArea { get; set; }
}