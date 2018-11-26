(to be completed)

* folders organization :
	- ../ext/sounds/
	- ../ext/data/
	-  ./python/

* Required Python package : - tensorflow : pip3 install tensorflow
* Use python3
* folders :
	- sounds : ../ext/sounds/NameYear_Spec
	- data   : ../ext/data/NameYear_Spec

		Name : name of the first author of the corresponding article
		Year : year of the article
		Spec : specificities, e.g. Lakatos2000_Perc

	!!! BE CAREFUL THE FOLDERS NAMES ARE CASE SENSITIVE !!!

* Files :
	- run optimisations : run_optims.py run optimisations for the representations and dataset defined in variables 'tsp' and 'rs'

	rs : TYPE_REPRESENTATION => type : {auditory, fourier} | representation : {spectrum, spectrogram, mps, strf}

	tsp : NameYear_Spec as for the folder names