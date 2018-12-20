%
%      thesis.cls
% 
%  Class for writing a thesis, based on book.cls
%  Hans E. Plesser, Goettingen, 1999
%
%
%  OPTIONS:
%
%  alternative:
%     dfinal       final thing
%     dlayout      final, but black boxes for overwide lines
%     ddate        final, but with date
%     dproof       wide margins, line skips, date
%
%  independent: 
%     showtags     show \label's, \rem's, \think's
%     nofigs       turn off figures
%     pdf          activate hyperref, switch to OT1 encoding
%                    to allow for PDF creation with VTeX
%                    provide \forcehyperanchor, set needsleveloneps
%
%  COMMANDS:
%
%  \clearchapter         Clear everything at end of chapter in such
%                        a way that left-over floats are put on pages
%                        with headers, but an added, empty even page
%                        remains completely empty
%
%  \clearemptydoublebage Create completely empty page at end of chapter
%
%  \nextfloatevenpage    Place next float on an even page (hope so ...)
%
%  \continuecaption[]{}  If caption too long for one page, remainder
%                        can be put on bottom of following page like
%                        a footnote; place IMMEDIATELY after
%                        \end{figure}.  Optional arg see \rightcaption.
%  
%  \rightcaption[]{}     Create a figure caption without creating an entry to
%                        the list of figures or advancing the figure
%                        counter.  Figure number will be that of
%                        preceeding figure.  If optional entry is
%                        missing, caption begins with 
%                            Figure X.XX(continued):
%                        otherwise with
%                            Figure X.XX#1:
%
%  \sublabel{}           Permit referencing to Fig. 1(a) ...
%                        to use: \caption[]{}
%                                \label{whole_fig}
%                                \sublabel{part_a}\sublabel{part_b}...
%                        reference as \ref{whole_fig} for whole, 
%                                     \ref{part_a} for part
%
%  \rem{}                put remark on margin, with showtags only
%
%  \think{}              put longer, boxed remark in middle of text,
%                        with showtags only
%
%  \longpage             lengthen/shorten page by
%  \shortpage             +-\baselineskip
%
%  \forcehyperanchor     hyperref does not notice References, this
%                          command forces recognition
%
%  needsleveloneps       Boolean to test if PS1 or PS2 version of
%                          figure should be used; true if PDF
%
%  NOTE: this class is by default set up for american English, with
%        German as a second language; if you want other languages,
%        search for "german", "american" below and replace!

%   Copyright (C) 2001 Hans Ekkehard Plesser
%
%   This program is free software; you can redistribute it and/or
%   modify it under the terms of the GNU General Public License
%   as published by the Free Software Foundation; either version 2
%   of the License, or (at your option) any later version.
%
%   This program is distributed in the hope that it will be useful,
%   but WITHOUT ANY WARRANTY; without even the implied warranty of
%   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%   GNU General Public License for more details.
%
%   You should have received a copy of the GNU General Public License
%   along with this program; if not, write to the Free Software
%   Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.

%%%%%%%%%%%%%%%%%% IDENTIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make sure we require a Latex not older than the one used
% in development
\NeedsTeXFormat{LaTeX2e}[1996/06/01]
\ProvidesClass{thesis}[1999/03/11 Thesis Class by Hans Ekkehard Plesser]

%%%%%%%%%%%%%%%%%% INITIALIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load packages that will be needed in diss.cls
\RequirePackage{calc}[1995/04/10]
\RequirePackage{ifthen}

%%%%%%%%%%%%%%%%%% DECLARE OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% de-activate standard draft and final options, 
% as they cause confusion
\DeclareOption{draft}{\OptionNotUsed}
\DeclareOption{final}{\OptionNotUsed}

% final: final layout, no date, no keys etc
\DeclareOption{dfinal}{\def\diss@layout{FINAL}%
                       \def\diss@date{HIDE}%
                       \def\diss@tags{HIDE}}

% Layout control, like final, but black boxes for overwide lines
\DeclareOption{dlayout}{\def\diss@layout{FINAL}%
                        \def\diss@date{HIDE}%
                        \def\diss@tags{HIDE}%
                        \PassOptionsToClass{draft}{book}}

% Final layout, dated
\DeclareOption{ddate}{\def\diss@layout{FINAL}%
                      \def\diss@date{SHOW}%
                      \def\diss@tags{HIDE}}

% Proof layout for others, wide margins and spacing, data
\DeclareOption{dproof}{\def\diss@layout{PROOF}%
                       \def\diss@date{SHOW}%
                       \def\diss@tags{HIDE}}

% \labels, \thinks, \rems are switched independent of layout option
\DeclareOption{showtags}{\def\diss@tags{SHOW}}

% EPS figures are switched independently of layout option
\DeclareOption{nofigs}{\PassOptionsToPackage{draft}{graphicx}}

% PDF generation requires some attention
\newboolean{vtex@pdf}           % new Booleans are false by def
\newboolean{needsleveloneps}
\DeclareOption{pdf}{\setboolean{vtex@pdf}{true}%
                    \setboolean{needsleveloneps}{true}}

% pass unused options to book
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}


%%%%%%%%%%%%%%%%%% EXECUTE OPTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% default is ddate
\ExecuteOptions{ddate}
\ProcessOptions\relax


%%%%%%%%%%%%%%%%%% LOAD BASE CLASS & PACKAGES %%%%%%%%%%%%%%%%%%

\LoadClass[a4paper]{book}
\usepackage[utf8]{inputenc}
\RequirePackage[norsk,american]{babel} % need some Norwegian
\RequirePackage{fancyheadings}           % flexible headings
\ifthenelse{\boolean{vtex@pdf}}%         % font encoding
  {\RequirePackage[OT1]{fontenc}}%       % VTeX requires OT1
  {\RequirePackage[T1]{fontenc}}         % T1 is better
\RequirePackage{amsmath}                 % powerful AMS LaTeX 
\RequirePackage{amssymb}               
\RequirePackage{chicago}                 % Chicago style citing
\RequirePackage{float}                   % added flexibility for floats
\RequirePackage{array}                   % better arrays, tabulars
\RequirePackage[figuresright]{rotating}  % landscape figures
\RequirePackage{longtable}               % multi-page tables
\RequirePackage{dcolumn}                 % fancy math table formatting
\RequirePackage{afterpage}        
\RequirePackage[footnotesize,bf]{caption}     % bold figure/table numbers
\ifthenelse{\equal{\diss@tags}{SHOW}}%   % show labels
   {\RequirePackage[notref,notcite]{showkeys}}%
   {\relax}
\ifthenelse{\boolean{vtex@pdf}}%
   {\RequirePackage[hypertexnames=false]{hyperref}}% % hyperrefs for PDF
   {\relax}

\usepackage{setspace}
\usepackage{graphicx}

\usepackage[labelformat=simple]{subcaption}
% change subref style of subcaptions to upper-case.
\renewcommand\thesubfigure{\Alph{subfigure}}

\usepackage[group-separator={,}]{siunitx}
\usepackage{geometry}
\usepackage{multicol}
\usepackage{lscape}
\usepackage{color}
\usepackage{colortbl}
\definecolor{LinkBlue}{rgb}{0,0.3,0.8}
\definecolor{LightGreen}{rgb}{0.75,1,0.75}
\definecolor{LightRed}{rgb}{1,0.7,0.7}
\definecolor{DarkGray}{gray}{0.3}
\definecolor{LightGray}{gray}{0.9}

\usepackage{float}
\restylefloat{table}

\usepackage{bm}
\usepackage[colorlinks=true, allcolors=LinkBlue]{hyperref}
\usepackage{tabularx}
\usepackage{tabu}
\usepackage{multirow}
\usepackage[usenames,dvipsnames]{xcolor}
\usepackage{listings}
\lstset{
		language=Python,
		keywords={Simulate, Create, CreateLayer, PlotLayer, PlotTargets, FindCenterElement, Connect, ConnectLayers, ConvergentConnect, RandomConvergentConnect, RandomDivergentConnect, GetTargetNodes, Distance, Displacement, GetTargetPositions, GetConnections},
    keywordstyle=\color{BrickRed},
    basicstyle=\ttfamily,
    commentstyle={},
    columns=fullflexible,
    numbers=none,
    numberstyle=\color[rgb]{0.4,0.4,0.5},
    % frame=leftline,
    % framerule=5pt,
    % rulecolor=\color[rgb]{0.85,0.85,0.85},
    frame=single,
    % framerule=1pt,
    showstringspaces=false,
    breakatwhitespace,
    breaklines=true,
    backgroundcolor=\color[rgb]{0.97, 0.97, 0.97},
    keepspaces=true
  }
\def\inline{\lstinline[keywords={}]}
\lstdefinelanguage{diff}{
  morecomment=[f][\color{red}]<,         % deleted lines 
  morecomment=[f][\color{NavyBlue}]>,       % added lines
}


%%%%%%%%%%%%%%%%%% MAIN PART %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% some status information to be given before starting on document
\AtEndOfClass{%
  \typeout{*********** Layout: \diss@layout  ***********}
  \typeout{*********** Date  : \diss@date    ***********}
  \typeout{*********** Keys  : \diss@tags    ***********}
  \ifthenelse{\boolean{vtex@pdf}}%
     {\typeout{*********** PDF generation / VTeX ***********}}%
     {\relax}
}

% make sure we start in English
\selectlanguage{american}

% Set page layout
\setlength{\topmargin}{4mm}
\setlength{\oddsidemargin}{5mm}
\setlength{\evensidemargin}{14.2mm}
\setlength{\textwidth}{140mm}
\setlength{\textheight}{220mm}
\setlength{\marginparwidth}{15mm}

% increase headheight to accomodate fancy headers
\addtolength{\headheight}{2.5pt}

% minor changes to tables
\setlength{\extrarowheight}{2pt}
\setlength{\arrayrulewidth}{0.4pt}
\setlength{\doublerulesep}{0pt}
\setlength{\tabcolsep}{6pt}
\renewcommand{\arraystretch}{1.05}

% Parameters for floats must be set at beginning of document
% to override defaults!
\AtBeginDocument{%
   \setcounter{bottomnumber}{1}            % for continued captions
   \setcounter{topnumber}{1}               % do not stack floats
   \setcounter{totalnumber}{2}             % do not stack floats
   \renewcommand{\textfraction}{0.1}       % allow large floats with text
   \renewcommand{\topfraction}{0.9}        % allow a lot at top
   \renewcommand{\bottomfraction}{0.4}     % allow continued captions
   \renewcommand{\floatpagefraction}{0.7}  % avoid empty pages
   \floatplacement{figure}{tp}             % no floats at bottom unless
   \floatplacement{table}{tp}              %  explicitly requested
}

% from LaTeX Begleiter, p. 100
\newcommand{\longpage}{\enlargethispage{\baselineskip}}
\newcommand{\shortpage}{\enlargethispage{-\baselineskip}}

% define command to produce empty left pages at end of chapter
% from LaTeX-Begleiter, p. 93
\newcommand{\clearemptydoublepage}{\newpage{\pagestyle{empty}\cleardoublepage}}

% clear at end of chapter, make sure floats come on pages with headers
\newcommand{\clearchapter}{\clearpage\clearemptydoublepage}


% place next float on even page
\newcommand{\nextfloatevenpage}{%
  \afterpage{%
     \clearpage%           get rid of pending floats after current page
     \ifthenelse{\isodd{\value{page}}}%  
        {\suppressfloats}%     no float on new, odd page
        {\relax}%
     }%
}

% right part of figure should have caption, but no LOF entry
\newcommand{\rightcaption}[2][(continued)]{%
 {\@makecaption{\figurename~\thefigure{}#1}{#2}}}

% continue caption on bottom of following page
\newcommand{\continuecaption}[2][(continued)]{%
  \begin{figure}[!b]
     \footnoterule
     \rightcaption[#1]{#2}    % no entry to figure list, no advanced counter
  \end{figure}}

% define sublabel command for referencing subfigures
% thanks to Steven D Cochran
\newcounter{sublabel}[figure]
\newcounter{ssublabel}[figure]
\def\thesublabel{(\alph{sublabel})}
\let\thessublabel\thesublabel
\def\sublabel#1{\refstepcounter{sublabel}\label{#1}%
                \refstepcounter{ssublabel}\label{s#1}}
\def\p@sublabel{\thefigure}
\def\p@ssublabel{}

% Activate fancy headings, define headers; footer follow below
\pagestyle{fancyplain}

% chapter mark, only needed for twosided print
\renewcommand{\chaptermark}[1]{\markboth{#1}{}}

% section mark
\renewcommand{\sectionmark}[1]{\markright{\thesection\ #1}}

\lhead[\fancyplain{}{\bfseries\thepage}]%    [even]{odd}
      {\fancyplain{}{\bfseries\rightmark}}
\chead[\fancyplain{}{}]{\fancyplain{}{}}
\rhead[\fancyplain{}{\bfseries\leftmark}]%
      {\fancyplain{}{\bfseries\thepage}}

% enter only chapters and sections in TOC---subsecs as a try
\setcounter{tocdepth}{2}

% enumerate down to subsections
\setcounter{secnumdepth}{2}

% enumerate (i), (ii) ...
\renewcommand{\labelenumi}{(\roman{enumi})}

% auxiliary command for hyperref, creates anchors if not done automatically
%   only needed in PDF-mode, when hyperref is active
\ifthenelse{\boolean{vtex@pdf}}%
  {\newcounter{hyperref@anchor@counter}%
   \newcommand{\forcehyperanchor}{\refstepcounter{hyperref@anchor@counter}}}%
  {\newcommand{\forcehyperanchor}{\relax}}

% If wanted, put date into box and set footers appropriately,
%   otherwise footers are empty
%   Counters defined outside to avoid spurious errors
\newcounter{hour} \newcounter{minute}
\ifthenelse{\equal{\diss@date}{SHOW}}
  {
    % Box containing date and time of day, see LaTeX Begleiter, p. 490
    %  Make it a box, as \today and \time are fixed at start.
    \DeclareRobustCommand{\dateandtime}{%
      \setcounter{hour}{\time/60}%
      \setcounter{minute}{\time - \value{hour}*60}%
      % type out Date HH:MMh, adding leading zeros where needed 
      \today~%
      \ifthenelse{\value{hour} < 10}{0\thehour:}{\thehour:}%
      \ifthenelse{\value{minute} < 10}{0\theminute{}h}{\theminute{}h}%
      }   % end dateandtime

    \lfoot[\fancyplain{\scriptsize\dateandtime}%
                      {\scriptsize\dateandtime}]%
          {\fancyplain{}{}}
    \cfoot[\fancyplain{}{}]{\fancyplain{}{}}
    \rfoot[\fancyplain{}{}]%
          {\fancyplain{\scriptsize\dateandtime}%
                      {\scriptsize\dateandtime}}%
  }
  {
    \lfoot[\fancyplain{}{}]{\fancyplain{}{}}
    \cfoot[\fancyplain{}{}]{\fancyplain{}{}}
    \rfoot[\fancyplain{}{}]{\fancyplain{}{}}
  }
  

% If rems and thinks are to be shown, define appropriate
%   commands, otherwise set commads to \relax
%   Note: showkeys for showing labels is loaded above
\ifthenelse{\equal{\diss@tags}{SHOW}}
  {\DeclareRobustCommand{\rem}[1]{\marginpar{\fbox{\parbox{0.9\marginparwidth}
                        {\raggedright\scriptsize #1}}}}
   \DeclareRobustCommand{\think}[1]{\noindent\par\vspace{2ex}\par
                \centerline{\fbox{\parbox{0.8\linewidth}{\footnotesize #1}}}}
   \DeclareRobustCommand{\thale}[1]{\noindent\par\vspace{2ex}\par
                \centerline{\fbox{\fbox{\parbox{0.8\linewidth}%
                                               {Thale:\\\footnotesize #1}}}}}
  }
  {
   \DeclareRobustCommand{\rem}[1]{\relax}
   \DeclareRobustCommand{\think}[1]{\relax}
   \DeclareRobustCommand{\thale}[1]{\relax}
  }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% End of file `diss.cls'.
