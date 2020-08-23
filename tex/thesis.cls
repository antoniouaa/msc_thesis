 %%
 %% lincolncsthesis.cls
 %% Copyright 2019 Benjamin Williams <bwilliams@lincoln.ac.uk>
 %%
 %% A LaTeX template for typesetting research degree theses
 %% according to the University of Lincoln School of Computer Science
 %% thesis guidelines.
 %%
 %% It is worth noting that the main inspiration (and small snippets of code) 
 %% for this template was Cameron Gray's bangorcsthesis template. 
 %%
 %% -----------------------------------------
 %%
 %% This work may be distributed and/or modified under the
 %% conditions of the LaTeX Project Public License, either version 1.3
 %% of this license of (at your option) any later version.
 %% The latest version of this license is in
 %% http://www.latex-project.org/lppl.txt
 %% and version 1.3 or later is part of all distributions of LaTeX
 %% version 2005/12/01 or later.
 %%


% We're gonna be defining variables as long globals
% (because they might wanna split these up with extra lines)
\def\lxdef{\long\xdef}

% This class name
\def\@classname {thesis}



% METADATA AND OPTIONS
% --------------------

% Define thesis template version and other metadata
\gdef\@templateVersion{1.0.3}
\gdef\@templateDate{2019/10/03}
\gdef\@templateAuthor{Benjamin Williams}

% It provides this class
\ProvidesClass{\@classname}[\@templateDate \@templateVersion \@templateAuthor]

% We need LaTeX2e at least
\NeedsTeXFormat{LaTeX2e}

% Article class
\LoadClass[12pt]{report}





% PACKAGES
% --------


% Packages
\RequirePackage{amsmath}
\RequirePackage{ifthen}
\RequirePackage{xkeyval}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage[T1]{fontenc} % font types and character verification
\RequirePackage[UKenglish]{babel}
\RequirePackage[UKenglish]{isodate}
\RequirePackage[utf8]{inputenc}
\RequirePackage{xparse}
\RequirePackage{graphicx}
\RequirePackage{longtable}
\RequirePackage{blindtext}
\RequirePackage{parskip}
\RequirePackage{setspace}
\RequirePackage{geometry}
\RequirePackage{layout}
\RequirePackage{courier}
\RequirePackage{subcaption}
\RequirePackage{url}
\RequirePackage{hyperref}
\RequirePackage[table,figure]{totalcount}
\RequirePackage{appendix}
\RequirePackage{csquotes}
\RequirePackage{titlesec}
\RequirePackage{datetime2}
\RequirePackage{fancyhdr}
\RequirePackage{lmodern}




% FANCY DATE STUFF
% -----------------

\newcommand{\monthyeardate}{%
  \DTMenglishmonthname{\@dtm@month} \@dtm@year
}

\newcommand{\yeardate}{%
  \@dtm@year
}

% Another handy macro to test if there's an option
\def\ifClassOption [#1]#2 {
    \@ifclasswith{\@classname}{#1}{#2}{\empty}
}

% Another handy macro to test if there's an option + else
\def\ifClassOptionElse [#1]#2#3 {
    \@ifclasswith{\@classname}{#1}{#2}{#3}
}


% THESIS TEMPLATE VARIABLES
% -------------------------


% Define variables relevant to this thesis
\lxdef \@logo {logo.png}
\newcommand{\thesisLogoPath}[1]{\xdef\@logo{#1}}
\lxdef \@date {\monthyeardate}
\newcommand{\thesisSubmissionDate}[1]{\xdef\@date{#1}}
\lxdef \@degree {Master of Science}
\newcommand{\thesisDegree}[1]{\xdef\@degree{#1}}
\lxdef \@programme {Computational Finance}
\newcommand{\thesisProgramme}[1]{\xdef\@programme{#1}}
\lxdef \@schoolname {Centre for Computational Finance and Economic Agents}
\newcommand{\thesisSchool}[1]{\xdef\@schoolname{#1}}
\lxdef \@collegename {CSEE}
\newcommand{\thesisCollege}[1]{\xdef\@collegename{#1}}
\lxdef \@universityname {University of Essex}
\newcommand{\thesisUniversity}[1]{\xdef\@universityname{#1}}
\newcommand{\thesisSupervisor}[1]{\xdef\@supervisor{#1}}
\newcommand{\thesisSecondSupervisor}[1]{\xdef\@secondsupervisor{#1}}

% START OF THESIS
% --------------

\fancypagestyle{plain}{
    \renewcommand{\headrulewidth}{0pt}%
    \fancyhf{}%
    \fancyfoot[R]{\footnotesize\textbf{\thepage}}%    
}

\fancypagestyle{mainStyle}{
	\renewcommand{\headrulewidth}{0pt}%
	\fancyhf{}%
	\fancyfoot[R]{\footnotesize\emph{\leftmark} \footnotesize\textbf{\thepage}}%   
	
	\@ifundefined{@includeheaders}{
	
	}{
    	\fancyhead[R]{\textit{\footnotesize \@modulecode~\@modulename}}
    	\fancyfoot[L]{\footnotesize \@author \@studentnumber}
	}
}


\newcommand{\thesisBodyStart}{
    \cleardoublepage
    \pagenumbering{arabic} 
}


% TABLE OF CONTENTS, FIGURES AND TABLES
% ---------------------------

\newcommand{\thesisTOF}{
    \iftotalfigures
        \cleardoublepage
        \listoffigures
    \fi
}

\newcommand{\thesisTOT}{
    \iftotaltables
        \cleardoublepage
        \listoftables
    \fi
}

\newcommand{\thesisTables}{
    \cleardoublepage
    \renewcommand{\contentsname}{Table of Contents}
    \setcounter{tocdepth}{3}
    {\setstretch{1.25}
        \tableofcontents
    }
    
    \setlength{\abovedisplayskip}{1pt}
    \setlength{\belowdisplayskip}{15pt}
}


% ABSTRACT
% -------------------------

\renewenvironment{abstract}{
    \cleardoublepage
    \chapter*{\abstractname}
}



% ACKNOWLEDGEMENTS
% ----------------

\newenvironment{acknowledgements}
    {\cleardoublepage
     \chapter*{Acknowledgements}
    }




% AFTER TITLE PAGE BUT BEFORE THESIS CONTENT
% ------------------------------------------
\newcommand{\afterTitlePage}{

    % Set margins for the rest of the thesis. UoL guidelines 
    % state that: [left >= 40mm], [others >= 15mm]
    \newgeometry{
        nohead, 
        top=15mm,
        bottom=20mm,
        right=25mm,
        left=25mm,
        marginparwidth=0cm,
        marginparsep=0cm
    }

    % Set spacing and reset numbering
    \pagenumbering{roman}
            
    % Set double spacing (from cleanthesis.sty)
    \setstretch{1.5}
    \clubpenalty = 10000
    \widowpenalty = 10000
    \displaywidowpenalty = 10000
}

% TITLE PAGE
% -------------

% Print out submission text
\newcommand{\submissionStatement}{
    \setstretch{1.25}
    \@ifundefined{@submissiontext}{Submitted in partial satisfaction of the requirements for the \\Degree of \@degree \\ in\\ \@programme}{\@submissiontext}
}

\newcommand{\printSupervisor}{
    \footnotesize\begin{tabular}{rl}
        \@ifundefined{@supervisor}{}
        {\textit{Supervisor} & \@supervisor \\}
        \@ifundefined{@secondsupervisor}{}
        {\textit{Second Supervisor} & \@secondsupervisor \\}
    \end{tabular}
}

% The default title page: title, then logo, then author
\newcommand{\@titlepage}{
    \begin{titlepage}

    % Make the title
    \makeatletter
    \centering{\huge \@title \par} \vspace{1.5cm}
    
    % Make the logo
    
    \includegraphics[height=1.5in]{\@logo}\par 
    % Some space, then the author, then the school and university
    % name
    \vspace{1.5cm}
    {\LARGE \@author \@ifundefined{@studentnumber}{}{\\ \@studentnumber} \par}\vspace{0.45cm}
    {\footnotesize \@schoolname \\ \vspace{0.25cm} \@collegename \\  \vspace{0.75cm}  {\large \@universityname} \par}
    
    % Some space and the submission statement
    \vspace{1cm}
    {\footnotesize \submissionStatement \par}
    
    % Print out the supervisor
    \vspace{1cm}
    \printSupervisor
    
    % Then the fancy date!
    \vspace{1cm}
    {\footnotesize \@ifundefined{@modulecode}{}{\@modulecode \\} August 2020 \par}
    
	\end{titlepage}
}

\newcommand{\beforeTitlePage}{
    % Set margins for the rest of the thesis
    % state that: [left >= 40mm], [others > 15mm]
    \newgeometry{
        nohead, 
        top=40mm,
        bottom=25mm,
        right=30mm,
        left=40mm,
        marginparwidth=0cm,
        marginparsep=0cm
    }
}

\def \@defaultTitlePage {\@titlepage}

\RenewDocumentCommand{\maketitle}{o}{
    \beforeTitlePage
    \@defaultTitlePage  
    \afterTitlePage
}


% TITLE / CHAPTERS / FIGURES / TITLESEC SPACING
% ---------------------------------------------

% Set the chapter title style
%\titleformat{\chapter}[display]
%{\chaptertitlename\ \thechapter}{0pt}{\LARGE}

\titleformat{\chapter}{\normalfont \bfseries \LARGE}{\thechapter}{11.5mm}{}

% And space it out a bit
\titlespacing*{\chapter} {0pt}{0pt}{0mm}

% A fake chapter title command to emulate chapter titles
% without actually adding a chapter
\newcommand{\chapterTitle}[1]{
	\vspace*{-20mm} 
	{\normalfont\LARGE\bfseries#1 \par} 
	\vspace*{20mm}
}


% REFERENCING
% -----------

% Fake harvard referencing, instead of using natbib -- because
% natbib makes biblatex blow up.
\def\@importHarvard {
    \RequirePackage[
        defernumbers=true,
        backend=biber,
        bibencoding=utf8,
        natbib=true,
        hyperref=true,
        backref=true,
        urldate=long,
        style=authoryear,
        citestyle=authoryear,
        sorting=nyt
    ]{biblatex}
}

% Otherwise use ieee's style and numeric citation
\def\@importBibLaTeX {
    \RequirePackage[
        defernumbers=true,
        backend=biber,
        bibencoding=utf8,
        natbib=true,
        hyperref=true,
        backref=true,
        urldate=long,
        style=ieee,
        citestyle=numeric,
        sorting=nyt
    ]{biblatex}
}

% Import harvard if [harvard] is specified, otherwise
% standard biblatex!
\ifClassOptionElse[harvard]{\@importHarvard}{\@importBibLaTeX}


% Command for printing out references list
\newcommand{\printReferences}[1][References]{
	
	% Set some spacing up between the items
	\setlength\bibitemsep{0.35cm}
	
	% Set the line height (of actual items) to 1.0
	\setstretch{1.0}
	
	% Print out the bibliography with the given title
	\printbibliography[title=#1]
}

\newcommand{\citesuper}[1]{%
    \textsuperscript{\cite{#1}}%
}%