
def correl_coef_composante(im1,im2):

    N = length(im1)

    moy_1 = mean(im1)
    moy_2 = mean(im2)

    Mat_cor = zeros(2)

    ec_1 = std(im1)
    ec_2 = std(im2)
    ec = [ec_1 ,ec_2 ]

    moy = [moy_1 ,moy_2 ]

    ima = [im1, im2 ]
    pass
