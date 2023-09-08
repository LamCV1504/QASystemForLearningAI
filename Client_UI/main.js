window.onload = function () {
  const URL_API = 'http://localhost:8000';
  const QA_URL = new URL(URL_API);

  const searchSection = document.querySelector('.section-search');
  const searchBox = document.querySelector('.search-box');
  const closeButton = document.querySelector('.close');
  const mainSection = document.querySelector('main');

  const createErrorParagraph = (msg) => {
    const paragraph = document.createElement('p');
    paragraph.className = 'card';
    paragraph.style.color = 'coral';
    paragraph.innerText = msg;
    return paragraph;
  };

  const removeAnswerSection = () => {
    const sectionAnswer = document.querySelector('.section-answer');
    if (sectionAnswer) {
      Object.assign(sectionAnswer.style, {
        translate: '0 100%',
        opacity: '0',
      });
      setTimeout(() => {
        mainSection.removeChild(sectionAnswer);
        searchSection.style.height = '100%';
      }, 500);
    }
  };

  Particles.init({
    selector: '.background',
    color: '#4b53b8',
    connectParticles: true,
  });

  const loadingSpinner = document.createElement('div');
  const input = document.createElement('input');

  loadingSpinner.className = 'lds-dual-ring';

  input.placeholder = 'Bạn thắc mắc điều gì về nhập môn AI?';
  input.setAttribute('spellCheck', 'false');

  input.addEventListener('click', function (e) {
    e.stopPropagation();
  });

  input.addEventListener('keydown', async function (e) {
    if (e.key === 'Enter') {
      if (this.value.trim() === '') {
        return;
      }

      removeAnswerSection();

      QA_URL.searchParams.append('question', this.value.trim());

      searchBox.removeChild(closeButton);
      searchBox.appendChild(loadingSpinner);
      Object.assign(searchBox.style, {
        pointerEvents: 'none',
        cursor: 'not-allowed',
      });

      const sectionAnswer = document.createElement('section');
      sectionAnswer.className = 'section-answer';

      try {
        const response = await fetch(QA_URL);
        const { answer, errorCode } = await response.json();

        if (errorCode) {
          sectionAnswer.appendChild(createErrorParagraph(answer));
        } else {
          const paragraphs = answer
            .map((e, i) => {
              return `<p class="card">
                  ${i + 1} ) ${e}
                </p>`;
            })
            .join('');

          sectionAnswer.innerHTML = paragraphs;
        }
      } catch (e) {
        sectionAnswer.appendChild(
          createErrorParagraph('Có lỗi xảy ra, vui lòng kiểm tra lại!'),
        );
        console.log(`Error; ${e}`);
      } finally {
        mainSection.appendChild(sectionAnswer);
        searchSection.style.height = 'max-content';
        searchBox.removeChild(document.querySelector('.lds-dual-ring'));
        searchBox.appendChild(closeButton);
        Object.assign(searchBox.style, {
          pointerEvents: 'auto',
          cursor: 'pointer',
        });
      }
    }
  });

  searchBox.addEventListener('click', function () {
    closeButton.classList.remove('hide');
    closeButton.classList.add('show');
    input.value = '';
    this.appendChild(input);
    this.classList.add('grow');
  });

  closeButton.addEventListener('click', function (e) {
    e.stopPropagation();
    this.classList.remove('show');
    this.classList.add('hide');
    searchBox.removeChild(document.querySelector('.search-box > input'));
    searchBox.classList.remove('grow');

    removeAnswerSection();
  });
};
