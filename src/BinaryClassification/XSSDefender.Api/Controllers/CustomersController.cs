using Microsoft.AspNetCore.Mvc;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Xml.Linq;

namespace XSSDefender.Api.Controllers;

[ApiController]
[Route("[controller]")]
public class CustomersController : ControllerBase
{
    PredictionEngine<XssInput, XssPrediction> predictionEngine;

    public CustomersController(PredictionEngine<XssInput, XssPrediction> predictionEngine)
    {
        this.predictionEngine = predictionEngine;
    }

    [HttpPost(Name = "CreateCustomer")]
    public IActionResult Post(Customer customer)
    {
        var input = new XssInput { Sentence = customer.Note };

        var prediction = predictionEngine.Predict(input);

        if (prediction.Prediction)
        {
            return BadRequest("XSS Attack detected!");
        }

        return Ok("Created");
    }
}


public class Customer
{
    public string Note { get; set; }
}


public class XssInput
{
    public string Sentence { get; set; }
}

public class XssPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
}