import cli.app
import PaintByNumbers.PictureGenerator as pgen


@cli.app.CommandLineApp
def GeneratePaintByNumber(app):
    print("Paint by numbers!")
    paint = pgen.PictureGenerator()
    paint.PerformKMeans(app.params.path, app.params.result, app.params.numColors)

#ls.add_param("-l", "--long", help="list in long format", default=False, action="store_true")
GeneratePaintByNumber.add_param("path", help="Path to picture", default="", type=str)
GeneratePaintByNumber.add_param("result", help="Path to result", default="", type=str)
GeneratePaintByNumber.add_param("numColors", help="Number of colors", default=5, type=int)

if __name__ == "__main__":
    #GeneratePaintByNumber.run()
    
    paint = pgen.PictureGenerator()
    paint.PerformKMeans(".\\images\\beth2.jpg", ".\\results\\", 40, False, True)