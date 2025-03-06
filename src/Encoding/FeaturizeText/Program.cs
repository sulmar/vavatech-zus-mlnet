
using Microsoft.ML;

Console.WriteLine("Hello, FeaturizeText!");

var context = new MLContext();

var data = new TextData[]
{

    new TextData { Text = "Kot biega po podwórku"},
    new TextData { Text = "Pies szczeka na kota"},
    new TextData { Text = "Ryba pływa w wodzie"},
};

var dataView = context.Data.LoadFromEnumerable(data);

var pipeline = context.Transforms.Text.FeaturizeText("Features", "Text");

var transformer = pipeline.Fit(dataView);
var transformedData = transformer.Transform(dataView);


var preview = transformedData.Preview();


Console.ReadLine();


public class TextData
{
    public string Text { get; set; }
}

