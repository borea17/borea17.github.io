var all_slides = [
    ['conditional_joint_probability_calculation', 'How can the conditional joint probability be calculated?'],
    ['test', 'new']
]

function write_down_all_slides(text_element) {
    console.log(all_slides);
    for (var i=0; i < all_slides.length; i++) {
        var question_text = (i+1).toString() + ': ' + all_slides[i][1] + '<br>';
        if (i + 1 < 10){
            num = '0' + (i + 1).toString();
        } else {
            num = i;
        }
        var h_ref_link = 'https://borea17.github.io/anki/probability_theory/id' + num + '/' + all_slides[i][0];

        newlink = document.createElement('a');
        newlink.innerHTML = question_text;
        newlink.setAttribute('title', question_text);
        newlink.setAttribute('href', h_ref_link);
        newlink.setAttribute('style', 'color: blue;');

        text_element.appendChild(newlink);
    };
}

