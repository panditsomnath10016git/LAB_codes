# LAB_codes
Codes used in lab for analysis and simulation

### Using "phc_codes" folder as a package in python
copy "phc_codes" folder to the python package site (in windows, usually at - `'C:\Users\somna\AppData\Local\Programs\Python\Python31x\Lib'`) to import from anywhere. Otherwise you can keep the folder in your working directory.

Now in any of your codes you can import the fuctions as import `phc_codes.xxx as pc`

### Plot formatting with matplotlib
Use your own customized template (`.mplstyle`) for plotting with matplotlib. Some examples are in folder 'plt_templates'. 
`plt.style.use('../plt_templates/line1.mplstyle')`