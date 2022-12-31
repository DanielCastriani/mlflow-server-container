ageInput = document.getElementById('Age')
yearsExperienceInput = document.getElementById('YearsExperience')

ageError = document.getElementById('AgeError')
yearsExperienceError = document.getElementById('YearsExperienceError')

resultText =  document.getElementById('Result')

submitBTN = document.getElementById('SubmitBTN')

SubmitBTN.addEventListener('click', (event)=> {
    event.preventDefault()

    ageError. value = ''
    yearsExperienceError. value = ''
    req = {   
        age: parseFloat(ageInput.value.replace(',', '.')),
        yearsExperience: parseFloat(yearsExperienceInput.value.replace(',', '.')),
    }

    submitBTN.disabled =  true
    fetch(
        '/predict_salary', {
            method: 'POST',
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(req)
        }
    )
    .then(res=> {
        console.log(res)
        res.json()
        .then((res)=> {
            if(res.success) {
                ageError.innerText = ''
                yearsExperienceError.innerText = ''

                resultText.innerText = `$ ${res.body.Salary.toFixed(2)}/Year`

            }else {
                ageError.innerText = !!res.Age ? res.Age : '';
                yearsExperienceError.innerText = !!res.YearsExperience ? res.YearsExperience : '';
            }

        })
    })
    .catch(res=> {
       
    })
    .finally(()=> {
        submitBTN.disabled =  false
    })

})
