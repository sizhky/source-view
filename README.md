# source-view
(Basic) Functionality to view a python library as a graph 

The main purpose of this utility is to view a library from bird's eye view without touching the codebase in anyway. Having a quick understanding of the dependecies and files will help one to get a feel of the code base.

## Usage

```bash
$ python source_view.py ~/Documents/github/torch_snippets/torch_snippets/`
```
This would generate a file called as `torch_snippets.html` at `~/Documents/github/torch_snippets/torch_snippets/`  
Download [torch_snippets.html](examples/torch_snippets.html) to see it interactive. This graph was generated for [torch-snippets](https://github.com/sizhky/torch-snippets)
library. Feedback solicited on your own libraries as well. I have not tested it on complex libraries.

![image](https://user-images.githubusercontent.com/3656100/143021113-ab6c40e6-7d48-4045-9a72-3f3612ab37a4.png)

Legend:
  * Red: File
  * Purple: Class
  * Green: Function
  * Yellow: Other functions/dependencies

## Performance

The bottleneck is to generate graph layout for the entire source. I have tested with a few libraries and is slow for medium size. For example it took 3m 17s to render graph for scrapy which contained 4239 nodes and 7600 edges.

IMO this is best for analyzing library with under 20 files.

I ran some more examples and have uploaded them in [examples](examples) folder
